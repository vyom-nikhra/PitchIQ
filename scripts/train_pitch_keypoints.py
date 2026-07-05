"""Train the pitch-keypoint heatmap net on SoccerNet-Calibration.

This is the upgrade path for the one documented calibration weakness
(box-only views with a single line family): a CNN that localises the pitch's
~30 semantic keypoints per frame, giving dense correspondences everywhere.

LEGAL: SoccerNet data is NDA-restricted (non-commercial research, no
redistribution). Keep the dataset and anything derived from it OUT of git and
OFF third-party services — train LOCALLY (an RTX 3050 handles this net).
Download first:
    python scripts/download_data.py --soccernet calibration

Then:
    python scripts/train_pitch_keypoints.py --epochs 30

SoccerNet-Calibration annotates *line extremities*, so keypoint supervision
is derived: named line×line intersections → our keypoint taxonomy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

# SoccerNet line-name pairs → PitchIQ keypoint names.
# NOTE SoccerNet taxonomy: 'Side line left/right' are the GOAL lines,
# 'Side line top/bottom' the touchlines; 'Circle left/right' are penalty arcs.
INTERSECTIONS = {
    ("Side line top", "Side line left"): "corner_tl",
    ("Side line top", "Side line right"): "corner_tr",
    ("Side line bottom", "Side line left"): "corner_bl",
    ("Side line bottom", "Side line right"): "corner_br",
    ("Middle line", "Side line top"): "halfway_top",
    ("Middle line", "Side line bottom"): "halfway_bottom",
    ("Big rect. left main", "Big rect. left top"): "pa_left_front_top",
    ("Big rect. left main", "Big rect. left bottom"): "pa_left_front_bottom",
    ("Big rect. right main", "Big rect. right top"): "pa_right_front_top",
    ("Big rect. right main", "Big rect. right bottom"): "pa_right_front_bottom",
    ("Small rect. left main", "Small rect. left top"): "ga_left_front_top",
    ("Small rect. left main", "Small rect. left bottom"): "ga_left_front_bottom",
    ("Small rect. right main", "Small rect. right top"): "ga_right_front_top",
    ("Small rect. right main", "Small rect. right bottom"): "ga_right_front_bottom",
    # box lines meeting the goal lines
    ("Big rect. left top", "Side line left"): "pa_left_goal_top",
    ("Big rect. left bottom", "Side line left"): "pa_left_goal_bottom",
    ("Big rect. right top", "Side line right"): "pa_right_goal_top",
    ("Big rect. right bottom", "Side line right"): "pa_right_goal_bottom",
    ("Small rect. left top", "Side line left"): "ga_left_goal_top",
    ("Small rect. left bottom", "Side line left"): "ga_left_goal_bottom",
    ("Small rect. right top", "Side line right"): "ga_right_goal_top",
    ("Small rect. right bottom", "Side line right"): "ga_right_goal_bottom",
}

# vertical orientation disambiguation: SoccerNet 'top' lines are pitch-top in
# ITS convention; our keypoint names carry the same semantic, and the pitch's
# mirror symmetry is resolved downstream — consistency is what matters here.


def _fit_ellipse_pts(pts: np.ndarray):
    if len(pts) < 6:
        return None
    try:
        import cv2

        (cx, cy), (a1, a2), ang = cv2.fitEllipse(
            np.ascontiguousarray(pts.astype(np.float32)))
    except Exception:
        return None
    MA, ma = (a1, a2) if a1 >= a2 else (a2, a1)
    if a2 > a1:
        ang += 90.0
    return cx, cy, MA, ma, ang


def keypoints_from_annotation(ann: dict, w: int, h: int) -> dict[str, tuple[float, float]]:
    """Derive keypoint supervision from SoccerNet line/curve annotations.

    Line×line intersections are projectively exact. Circle-derived points
    use the fitted ellipse: halfway∩ellipse (circle top/bottom) and box-front
    ∩arc (arc keypoints) are exact-ish; ellipse centre (→ centre spot) and
    x-extremes (→ circle left/right) carry a small perspective bias that is
    acceptable as heatmap supervision (RANSAC absorbs it at solve time).
    """
    from pitchiq.core.geometry import line_intersection
    from pitchiq.perception.calibration.conics import (conic_line_intersections,
                                                       ellipse_to_conic, line_through)

    def pts_of(name):
        pts = ann.get(name)
        if not pts or len(pts) < 2:
            return None
        return np.array([[q["x"] * w, q["y"] * h] for q in pts])

    def line_of(name):
        p = pts_of(name)
        return None if p is None else (p[0], p[-1])

    def inside(pt, margin=0.2):
        return (-margin * w < pt[0] < (1 + margin) * w
                and -margin * h < pt[1] < (1 + margin) * h)

    def y_orientation():
        """True = SoccerNet-'top' is image-top in this frame; None = unknown."""
        t, b = pts_of("Side line top"), pts_of("Side line bottom")
        if t is not None and b is not None:
            return float(t[:, 1].mean()) < float(b[:, 1].mean())
        for name, expect_upper in (("Big rect. left top", "Big rect. left bottom"),
                                   ("Big rect. right top", "Big rect. right bottom")):
            u, v = pts_of(name), pts_of(expect_upper)
            if u is not None and v is not None:
                return float(u[:, 1].mean()) < float(v[:, 1].mean())
        return None

    def _first(*names):
        for n in names:
            p = pts_of(n)
            if p is not None:
                return p
        return None

    def x_orientation():
        """True = SoccerNet-'left' is image-left; None = unknown."""
        l = _first("Side line left", "Big rect. left main")
        r = _first("Side line right", "Big rect. right main")
        if l is not None and r is not None:
            return float(l[:, 0].mean()) < float(r[:, 0].mean())
        if l is not None:
            return float(l[:, 0].mean()) < w / 2
        if r is not None:
            return float(r[:, 0].mean()) > w / 2
        return None

    y_norm = y_orientation()
    x_norm = x_orientation()

    out = {}
    for (la, lb), kp_name in INTERSECTIONS.items():
        a = line_of(la)
        b = line_of(lb)
        if a is None or b is None:
            continue
        pt = line_intersection(a[0], a[1], b[0], b[1])
        if pt is not None and inside(pt):
            out[kp_name] = (float(pt[0]), float(pt[1]))

    # ---- centre circle ----------------------------------------------------
    circ = pts_of("Circle central")
    mid = line_of("Middle line")
    if circ is not None and len(circ) >= 8:
        fit = _fit_ellipse_pts(circ)
        if fit is not None:
            cx, cy, MA, ma, ang = fit
            if 0.04 * w < MA < 1.6 * w:
                if inside((cx, cy), 0.05):
                    out["center_spot"] = (float(cx), float(cy))
                # visible x-extremes of the annotated arc ≈ circle left/right —
                # only labelled when this frame's left/right orientation is known
                lo, hi = circ[np.argmin(circ[:, 0])], circ[np.argmax(circ[:, 0])]
                if hi[0] - lo[0] > 0.7 * MA and x_norm is not None:
                    lkey, rkey = ("circle_left", "circle_right") if x_norm else \
                                 ("circle_right", "circle_left")
                    out[lkey] = (float(lo[0]), float(lo[1]))
                    out[rkey] = (float(hi[0]), float(hi[1]))
                if mid is not None and y_norm is not None:
                    Q = ellipse_to_conic(cx, cy, MA, ma, ang)
                    cuts = conic_line_intersections(Q, line_through(mid[0], mid[1]))
                    if len(cuts) == 2:
                        top, bot = sorted(cuts, key=lambda p: p[1])
                        if inside(top) and inside(bot):
                            tkey, bkey = ("circle_top", "circle_bottom") if y_norm else \
                                         ("circle_bottom", "circle_top")
                            out[tkey] = (float(top[0]), float(top[1]))
                            out[bkey] = (float(bot[0]), float(bot[1]))

    # ---- penalty arcs ------------------------------------------------------
    for side in ("left", "right"):
        arc = pts_of(f"Circle {side}")
        front = line_of(f"Big rect. {side} main")
        if arc is None or front is None or len(arc) < 6:
            continue
        fit = _fit_ellipse_pts(arc)
        if fit is None:
            continue
        cx, cy, MA, ma, ang = fit
        Q = ellipse_to_conic(cx, cy, MA, ma, ang)
        cuts = conic_line_intersections(Q, line_through(front[0], front[1]))
        if len(cuts) == 2 and y_norm is not None:
            top, bot = sorted(cuts, key=lambda p: p[1])
            if inside(top) and inside(bot):
                tkey, bkey = (f"arc_{side}_top", f"arc_{side}_bottom") if y_norm else \
                             (f"arc_{side}_bottom", f"arc_{side}_top")
                out[tkey] = (float(top[0]), float(top[1]))
                out[bkey] = (float(bot[0]), float(bot[1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/soccernet/calibration")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="cap samples (debug)")
    ap.add_argument("--out", default="weights/pitch_keypoints.pt")
    args = ap.parse_args()

    import cv2
    import torch
    import torch.nn.functional as F

    from pitchiq.core.pitch import Pitch
    from pitchiq.intelligence import encoder as _  # noqa: F401 (torch check)
    from pitchiq.io.soccernet import iter_calibration_samples
    from pitchiq.perception.calibration.keypoints import KeypointCalibrator, build_keypoint_net

    pitch = Pitch()
    names, _arr = pitch.keypoint_array()
    name_idx = {n: i for i, n in enumerate(names)}
    W, H = KeypointCalibrator.INPUT_SIZE  # 480, 270

    samples = []
    for img_path, ann in iter_calibration_samples(args.data_dir, "train"):
        kps = keypoints_from_annotation(ann, W, H)
        kps = {k: v for k, v in kps.items() if k in name_idx}
        if len(kps) >= 4:
            samples.append((img_path, kps))
        if args.limit and len(samples) >= args.limit:
            break
    print(f"{len(samples)} training samples with >=4 derivable keypoints")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_keypoint_net(len(names)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def load_batch(batch):
        imgs, heats = [], []
        hh, hw = H // 2, W // 2  # the U-Net decodes back to half resolution
        for img_path, kps in batch:
            img = cv2.resize(cv2.imread(str(img_path)), (W, H))[:, :, ::-1] / 255.0
            imgs.append(img.transpose(2, 0, 1).astype(np.float32))
            heat = np.zeros((len(names), hh, hw), dtype=np.float32)
            yy, xx = np.mgrid[0:hh, 0:hw]
            for k, (x, y) in kps.items():
                cx, cy = x / 2.0, y / 2.0
                heat[name_idx[k]] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0**2))
            heats.append(heat)
        return (torch.from_numpy(np.stack(imgs)).to(device),
                torch.from_numpy(np.stack(heats)).to(device))

    def save_torchscript(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        model.eval()
        cpu_model = build_keypoint_net(len(names))
        cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
        cpu_model.eval()
        torch.jit.trace(cpu_model, torch.zeros(1, 3, H, W)).save(str(path))
        model.train()

    from concurrent.futures import ThreadPoolExecutor

    out = Path(args.out)
    rng = np.random.default_rng(0)
    pool = ThreadPoolExecutor(max_workers=2)
    pos_w = torch.tensor(40.0, device=device)
    for ep in range(args.epochs):
        rng.shuffle(samples)
        batches = [samples[i: i + args.batch] for i in range(0, len(samples), args.batch)]
        losses = []
        # prefetch: CPU decodes the next batch while the GPU trains on this one
        future = pool.submit(load_batch, batches[0])
        for bi in range(len(batches)):
            x, y = future.result()
            if bi + 1 < len(batches):
                future = pool.submit(load_batch, batches[bi + 1])
            pred = model(x)
            loss = F.binary_cross_entropy_with_logits(pred, y, pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        print(f"epoch {ep + 1}/{args.epochs}: loss {np.mean(losses):.4f}", flush=True)
        save_torchscript(out.with_suffix(".ckpt.pt"))  # crash-safe checkpoint
    pool.shutdown()

    save_torchscript(out)
    print(f"saved {out} - set calibration.keypoint_weights to use it.\n"
          "REMINDER: weights trained on NDA data stay out of the public repo "
          "by default (see docs/data_sources.md).", flush=True)


if __name__ == "__main__":
    main()
