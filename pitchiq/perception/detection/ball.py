"""Dedicated ball strategy: ROI re-inference, motion gating, gap interpolation.

The ball is the least reliable perception target (small, motion-blurred,
occluded, airborne). Three mitigations, all documented in the README:

1. **ROI second pass** — when the full-frame detector misses the ball, re-run
   detection at high resolution on a crop around the ball's predicted position.
2. **Motion gating** — a constant-velocity Kalman filter picks the candidate
   nearest the prediction and rejects teleports.
3. **Temporal interpolation** — post-hoc, gaps up to ``max_gap_interpolate``
   frames are bridged by linear interpolation with decayed confidence.

Airborne balls still project incorrectly through the ground-plane homography
(a known, documented limitation of single-camera systems).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitchiq.config import BallConfig
from pitchiq.core.types import Detection, EntityClass


class BallKalman:
    """Constant-velocity KF on the ball's pixel center (state: x, y, vx, vy)."""

    def __init__(self, q: float = 2.0, r: float = 4.0) -> None:
        self.x: np.ndarray | None = None
        self.P = np.eye(4) * 50.0
        self.q = q
        self.r = r

    def predict(self) -> np.ndarray | None:
        if self.x is None:
            return None
        F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.x = F @ self.x
        Q = np.diag([self.q, self.q, self.q * 2, self.q * 2])
        self.P = F @ self.P @ F.T + Q
        return self.x[:2].copy()

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=float)
        if self.x is None:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            return
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = np.eye(2) * self.r
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P


class BallSelector:
    """Stateful per-frame ball resolver combining detector output, ROI
    re-inference, and Kalman gating. Returns at most one ball detection."""

    def __init__(self, cfg: BallConfig, detector=None, max_jump_px: float = 120.0) -> None:
        self.cfg = cfg
        self.detector = detector  # needs .detect_roi for the second pass
        self.kf = BallKalman()
        self.miss_streak = 0
        self.max_jump_px = max_jump_px

    def select(self, frame_bgr: np.ndarray, detections: list[Detection]) -> Detection | None:
        balls = [d for d in detections if d.cls == EntityClass.BALL]
        pred = self.kf.predict()

        if not balls and self.cfg.roi_inference and pred is not None and self.detector is not None:
            balls = self._roi_pass(frame_bgr, pred)

        best = self._pick(balls, pred)
        if best is None:
            self.miss_streak += 1
            if self.miss_streak > 25:  # stale prediction: reset filter
                self.kf = BallKalman()
            return None
        self.miss_streak = 0
        self.kf.update(np.array(best.center))
        return best

    def _pick(self, balls: list[Detection], pred: np.ndarray | None) -> Detection | None:
        if not balls:
            return None
        if pred is None:
            return max(balls, key=lambda d: d.conf)
        scored = []
        for d in balls:
            dist = float(np.hypot(d.center[0] - pred[0], d.center[1] - pred[1]))
            if dist > self.max_jump_px * (1 + 0.5 * self.miss_streak):
                continue
            scored.append((dist - 50.0 * d.conf, d))
        if not scored:
            return None
        return min(scored, key=lambda t: t[0])[1]

    def _roi_pass(self, frame_bgr: np.ndarray, pred: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        half = self.cfg.roi_size // 2
        cx = int(np.clip(pred[0], half, max(half, w - half)))
        cy = int(np.clip(pred[1], half, max(half, h - half)))
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0 or not hasattr(self.detector, "detect_roi"):
            return []
        found = self.detector.detect_roi(crop, conf=self.cfg.conf_threshold)
        out = []
        for d in found:
            if d.cls != EntityClass.BALL:
                continue
            bb = d.bbox.copy()
            bb[[0, 2]] += x1
            bb[[1, 3]] += y1
            out.append(Detection(bbox=bb, conf=d.conf * 0.9, cls=EntityClass.BALL))
        return out


def interpolate_ball(df: pd.DataFrame, max_gap: int, ball_id: int = -1) -> pd.DataFrame:
    """Bridge missing-ball gaps in the tracking table by linear interpolation.

    Interpolated rows get decayed confidence (0.5 * linear taper) so consumers
    can distinguish observed from inferred ball positions.
    """
    ball = df[df["entity_id"] == ball_id].sort_values("frame")
    others = df[df["entity_id"] != ball_id]
    if ball.empty:
        return df
    all_frames = np.arange(int(df["frame"].min()), int(df["frame"].max()) + 1)
    have = ball["frame"].to_numpy()
    fps_dt = np.median(np.diff(df["timestamp"].sort_values().unique())) if len(df) else 0.04

    new_rows = []
    for prev_f, next_f in zip(have[:-1], have[1:]):
        gap = int(next_f - prev_f)
        if gap <= 1 or gap > max_gap:
            continue
        row_a = ball.loc[ball["frame"] == prev_f].iloc[0]
        row_b = ball.loc[ball["frame"] == next_f].iloc[0]
        for f in range(int(prev_f) + 1, int(next_f)):
            t = (f - prev_f) / gap
            taper = 1.0 - abs(2 * t - 1)
            new = row_a.copy()
            new["frame"] = f
            new["timestamp"] = row_a["timestamp"] + (f - prev_f) * fps_dt
            for col in ("x_pixel", "y_pixel", "x_pitch", "y_pitch"):
                a, b = row_a[col], row_b[col]
                new[col] = a + t * (b - a) if np.isfinite(a) and np.isfinite(b) else np.nan
            new["conf"] = float(min(row_a["conf"], row_b["conf"])) * 0.5 * (0.5 + 0.5 * taper)
            new_rows.append(new)
    if not new_rows:
        return df
    out = pd.concat([others, ball, pd.DataFrame(new_rows)], ignore_index=True)
    del all_frames
    return out.sort_values(["frame", "entity_id"]).reset_index(drop=True)
