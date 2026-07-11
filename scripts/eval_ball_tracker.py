"""Evaluate trained TrackNet ball models on chosen SoccerNet sequences.

Two models trained with different train/val splits cannot be compared on
their own logged numbers (different validation clips). This script runs each
model over the SAME sequences — ideally ones neither model trained on — and
sweeps the peak-confidence threshold, reporting detection rate / false
positives / pixel error per (model, threshold).

Usage:
    python scripts/eval_ball_tracker.py \
        --models weights/ball_tracknet_real.pt weights/ball_tracknet_kaggle.pt \
        --data-dir data/soccernet/tracking/tracking/train \
        --seqs SNMOT-060 SNMOT-070 SNMOT-099 SNMOT-109 --stride 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from pitchiq.perception.detection.ball_dataset import (  # noqa: E402
    BallWindowDataset,
    list_ball_windows,
)


def load_model(path: str, device: str):
    import torch

    from pitchiq.perception.detection.tracknet import build_tracknet

    try:
        model = torch.jit.load(path, map_location=device).eval()
    except (RuntimeError, ValueError):
        model = build_tracknet(3)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval().to(device)
    return model


def collect_predictions(model, loader, device: str):
    """Per window: (peak_prob, pred_x, pred_y, has_ball, gt_x, gt_y)."""
    import torch

    rows = []
    with torch.no_grad():
        for xb, _, has, px in loader:
            xb = xb.to(device, non_blocking=True)
            prob = torch.sigmoid(model(xb))[:, 0].float()
            B, _, Ww = prob.shape
            flat = prob.reshape(B, -1)
            peak, arg = flat.max(dim=1)
            iy, ix = (arg // Ww).float(), (arg % Ww).float()
            for b in range(B):
                rows.append((float(peak[b]), float(ix[b]), float(iy[b]),
                             float(has[b]) > 0.5, float(px[b, 0]), float(px[b, 1])))
    return rows


def metrics_at(rows, thresh: float) -> dict:
    n_pos = n_neg = det = fp = 0
    errs = []
    for peak, x, y, has, gx, gy in rows:
        if has:
            n_pos += 1
            if peak >= thresh:
                det += 1
                errs.append(float(np.hypot(x - gx, y - gy)))
        else:
            n_neg += 1
            if peak >= thresh:
                fp += 1
    det_rate = det / max(n_pos, 1)
    fp_rate = fp / max(n_neg, 1)
    med = float(np.median(errs)) if errs else float("nan")
    return dict(det=det_rate, fp=fp_rate, med=med,
                score=det_rate - fp_rate - (min(med, 20.0) / 200.0 if errs else 0.1),
                n_pos=n_pos, n_neg=n_neg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data-dir", default="data/soccernet/tracking/tracking/train")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--thresholds", default="0.2,0.35,0.5,0.65,0.8")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from pitchiq.perception.detection.tracknet import TrackNetBall

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = REPO / args.data_dir
    windows = []
    for name in args.seqs:
        w = list_ball_windows(data_dir / name, stride=args.stride)
        if not w:
            raise SystemExit(f"no windows for {name} under {data_dir}")
        windows += w
        print(f"  {name}: {len(w)} windows", flush=True)
    ds = BallWindowDataset(windows, TrackNetBall.INPUT_SIZE, augment=False,
                           cache_dir=REPO / "data" / "soccernet" / "cache_ball_512x288")
    dl = DataLoader(ds, batch_size=args.batch, num_workers=2,
                    pin_memory=device == "cuda")

    threshes = [float(t) for t in args.thresholds.split(",")]
    for mpath in args.models:
        model = load_model(mpath, device)
        rows = collect_predictions(model, dl, device)
        n_pos = sum(1 for r in rows if r[3])
        print(f"\n== {mpath}  ({len(rows)} windows, {n_pos} with ball) ==")
        print(f"{'thresh':>7} {'det%':>7} {'fp%':>7} {'med_px':>7} {'score':>7}")
        for t in threshes:
            m = metrics_at(rows, t)
            print(f"{t:7.2f} {100 * m['det']:7.1f} {100 * m['fp']:7.1f} "
                  f"{m['med']:7.2f} {m['score']:7.3f}")


if __name__ == "__main__":
    main()
