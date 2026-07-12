"""Optional pose sampling: body-shape descriptors per tracked player.

Runs an ultralytics pose model (COCO-17 keypoints) on sampled frames during
the Layer-1 pass, matches skeletons to tracks by box IoU, and reduces each
skeleton to a handful of scale-free body-shape descriptors (lean, stride,
arm spread, crouch, compactness). Per-track mean/std of these land in
``pose.parquet``, which Layer 3 appends to the style embeddings as a
``pose`` feature group — so "how a player carries themselves" contributes
to role discovery and similar-player search, per the peer survey (datum
uses ViTPose for the same purpose; we use yolo11-pose because it is tiny,
already in our stack, and runs on a 4 GB GPU alongside the detector).

Everything degrades gracefully: no ultralytics / no weights / no GPU means
no pose artifact, and downstream simply proceeds without the group.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# COCO-17 indices
_L_SHO, _R_SHO = 5, 6
_L_WRI, _R_WRI = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_ANK, _R_ANK = 15, 16


def pose_descriptor(kps: np.ndarray, box: np.ndarray,
                    min_conf: float = 0.3) -> np.ndarray | None:
    """(17, 3) keypoints [x, y, conf] + xyxy box -> 5 scale-free descriptors.

    Returns None when the core joints (shoulders/hips) are not confidently
    visible — a half-occluded skeleton would poison the per-track statistics.
    """
    kps = np.asarray(kps, dtype=float)
    h = max(float(box[3] - box[1]), 1e-6)
    conf = kps[:, 2]
    core = [_L_SHO, _R_SHO, _L_HIP, _R_HIP]
    if np.any(conf[core] < min_conf):
        return None
    sho = (kps[_L_SHO, :2] + kps[_R_SHO, :2]) / 2
    hip = (kps[_L_HIP, :2] + kps[_R_HIP, :2]) / 2
    torso = hip - sho
    lean = float(abs(np.arctan2(torso[0], torso[1] + 1e-9)))  # 0 = upright

    def _pair(a: int, b: int) -> float | None:
        if conf[a] < min_conf or conf[b] < min_conf:
            return None
        return float(np.linalg.norm(kps[a, :2] - kps[b, :2]))

    stride = _pair(_L_ANK, _R_ANK)
    stride = (stride / h) if stride is not None else np.nan
    arms = _pair(_L_WRI, _R_WRI)
    sho_w = _pair(_L_SHO, _R_SHO) or 1e-6
    arm_spread = (arms / max(sho_w, 1e-6)) if arms is not None else np.nan
    ank_ok = conf[_L_ANK] >= min_conf and conf[_R_ANK] >= min_conf
    if ank_ok:
        ank = (kps[_L_ANK, :2] + kps[_R_ANK, :2]) / 2
        crouch = float(np.linalg.norm(ank - hip)) / h  # small = crouched
    else:
        crouch = np.nan
    vis = kps[conf >= min_conf, :2]
    compact = float(np.linalg.norm(vis - vis.mean(0), axis=1).mean()) / h
    return np.array([lean, stride, arm_spread, crouch, compact], dtype=float)


_NAMES = ["lean", "stride", "arm_spread", "crouch", "compact"]


class PoseSampler:
    """Accumulates per-track pose descriptors over sampled frames."""

    def __init__(self, model_name: str = "yolo11n-pose.pt",
                 device: str = "auto", min_kp_conf: float = 0.3) -> None:
        self.min_kp_conf = min_kp_conf
        self.acc: dict[int, list[np.ndarray]] = {}
        self.model = None
        self.device = None if device == "auto" else device
        try:
            from ultralytics import YOLO

            self.model = YOLO(model_name)
            log.info("pose sampler ready: %s", model_name)
        except Exception as exc:  # no ultralytics / download blocked
            log.warning("pose sampling unavailable (%s)", exc)

    @property
    def ok(self) -> bool:
        return self.model is not None

    def sample(self, frame_bgr: np.ndarray, tracks) -> None:
        """Top-down pose: the model runs on each track's CROP, not the full
        frame — wide broadcast players are ~90 px tall, far below what
        full-frame pose detection resolves. We already know where every
        player is; the model only has to read the body shape."""
        if self.model is None or not tracks:
            return
        h, w = frame_bgr.shape[:2]
        crops, keep = [], []
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            mx, my = 0.2 * (x2 - x1), 0.08 * (y2 - y1)
            xa, ya = int(max(0, x1 - mx)), int(max(0, y1 - my))
            xb, yb = int(min(w, x2 + mx)), int(min(h, y2 + my))
            if xb - xa < 16 or yb - ya < 32:
                continue
            crops.append(np.ascontiguousarray(frame_bgr[ya:yb, xa:xb]))
            keep.append(t)
        if not crops:
            return
        try:
            results = self.model.predict(crops, verbose=False, conf=0.25,
                                         imgsz=256, device=self.device)
        except Exception as exc:
            log.warning("pose inference failed (%s); disabling", exc)
            self.model = None
            return
        for t, crop, res in zip(keep, crops, results):
            if res.keypoints is None or len(res.boxes) == 0:
                continue
            # largest skeleton in the crop = the tracked player
            areas = ((res.boxes.xyxy[:, 2] - res.boxes.xyxy[:, 0])
                     * (res.boxes.xyxy[:, 3] - res.boxes.xyxy[:, 1]))
            pi = int(areas.argmax())
            kps = res.keypoints.data.cpu().numpy()[pi]
            box = res.boxes.xyxy.cpu().numpy()[pi]
            desc = pose_descriptor(kps, box, self.min_kp_conf)
            if desc is not None:
                self.acc.setdefault(t.track_id, []).append(desc)

    def finalize(self, min_samples: int = 3) -> pd.DataFrame:
        """Per-track mean/std of each descriptor (players with too few clean
        skeletons are omitted rather than reported on thin evidence)."""
        rows = []
        for tid, descs in self.acc.items():
            D = np.stack(descs)
            if len(D) < min_samples:
                continue
            row: dict = {"entity_id": int(tid), "n_samples": int(len(D))}
            for i, n in enumerate(_NAMES):
                col = D[:, i]
                col = col[np.isfinite(col)]
                if len(col) == 0:
                    row[f"{n}_mean"], row[f"{n}_std"] = 0.0, 0.0
                else:
                    row[f"{n}_mean"] = float(np.mean(col))
                    row[f"{n}_std"] = float(np.std(col))
            rows.append(row)
        return pd.DataFrame(rows)
