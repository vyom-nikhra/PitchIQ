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

# SoccerNet line-name pairs → PitchIQ keypoint names (subset; extend freely)
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
}


def keypoints_from_annotation(ann: dict, w: int, h: int) -> dict[str, tuple[float, float]]:
    """Derive keypoint pixels from SoccerNet line annotations."""
    from pitchiq.core.geometry import line_intersection

    def line_of(name):
        pts = ann.get(name)
        if not pts or len(pts) < 2:
            return None
        p = np.array([[q["x"] * w, q["y"] * h] for q in pts])
        return p[0], p[-1]

    out = {}
    for (la, lb), kp_name in INTERSECTIONS.items():
        a = line_of(la)
        b = line_of(lb)
        if a is None or b is None:
            continue
        pt = line_intersection(a[0], a[1], b[0], b[1])
        if pt is not None and -0.2 * w < pt[0] < 1.2 * w and -0.2 * h < pt[1] < 1.2 * h:
            out[kp_name] = (float(pt[0]), float(pt[1]))
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
        hh, hw = H // 4, W // 4
        for img_path, kps in batch:
            img = cv2.resize(cv2.imread(str(img_path)), (W, H))[:, :, ::-1] / 255.0
            imgs.append(img.transpose(2, 0, 1).astype(np.float32))
            heat = np.zeros((len(names), hh, hw), dtype=np.float32)
            yy, xx = np.mgrid[0:hh, 0:hw]
            for k, (x, y) in kps.items():
                cx, cy = x / 4.0, y / 4.0
                heat[name_idx[k]] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0**2))
            heats.append(heat)
        return (torch.from_numpy(np.stack(imgs)).to(device),
                torch.from_numpy(np.stack(heats)).to(device))

    rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        rng.shuffle(samples)
        losses = []
        for i in range(0, len(samples), args.batch):
            x, y = load_batch(samples[i: i + args.batch])
            pred = model(x)
            loss = F.binary_cross_entropy_with_logits(pred, y, pos_weight=torch.tensor(40.0, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        print(f"epoch {ep + 1}/{args.epochs}: loss {np.mean(losses):.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    example = torch.zeros(1, 3, H, W)
    torch.jit.trace(model, example).save(str(out))
    print(f"saved {out} — set calibration.keypoint_weights to use it.\n"
          "REMINDER: weights trained on NDA data stay out of the public repo "
          "by default (see docs/data_sources.md).")


if __name__ == "__main__":
    main()
