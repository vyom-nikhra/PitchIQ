"""Train the TrackNet-style heatmap ball tracker.

Two data sources (mirrors the keypoint model's synthetic/real split):

* ``--source synthetic`` (default) — render simulated matches; the renderer
  knows the exact ball pixel every frame, so training data is unlimited and
  licence-clean. A model trained here is for the *synthetic* domain (the
  bundled demo), just as the synthetic renderer differs from real broadcast.
* ``--source soccernet`` — real ball ground truth from SoccerNet tracking
  (MOT gt.txt, class ball). This is what a real-broadcast ball tracker needs;
  requires the SoccerNet tracking download (NDA — local only, never commit).

The network regresses a Gaussian ball heatmap from 3 consecutive frames.

Usage:
    python scripts/train_ball_tracker.py --matches 4 --epochs 20
    python scripts/train_ball_tracker.py --source soccernet \
        --data-dir data/soccernet/tracking/train --epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402


def synthetic_windows(n_matches: int, half_minutes: float, w: int, h: int):
    """Yield (3-frame stack float [9,H,W], ball_pixel or None) from renders."""
    import cv2

    from pitchiq.config import load_config
    from pitchiq.core.pitch import Pitch
    from pitchiq.core.schema import BALL_ID
    from pitchiq.demo.render import render_match
    from pitchiq.demo.simulate import simulate_demo_match

    tmp = REPO / "data" / "jobs" / "_ball_train"
    tmp.mkdir(parents=True, exist_ok=True)
    for m in range(n_matches):
        cfg = load_config(overrides={"simulator": {"half_minutes": half_minutes,
                                                   "seed": 200 + m}})
        sim = simulate_demo_match(cfg.simulator)
        vid = tmp / f"m{m}.mp4"
        rr = render_match(sim.tracking, Pitch(), sim.meta.kit_colors, vid, cfg.simulator.fps,
                          width=w * 2, height=h * 2)
        ball = {int(r["frame"]): r for _, r in
                rr.boxes[rr.boxes.entity_id == BALL_ID].iterrows()}
        cap = cv2.VideoCapture(str(vid))
        frames = []
        fidx = -1
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fidx += 1
            small = cv2.resize(fr, (w, h))[:, :, ::-1].astype(np.float32) / 255.0
            frames.append(small.transpose(2, 0, 1))
            if len(frames) > 3:
                frames.pop(0)
            if len(frames) == 3:
                b = ball.get(fidx)
                px = None
                if b is not None:
                    cx = (b["x1"] + b["x2"]) / 2 / (w * 2) * w
                    cy = (b["y1"] + b["y2"]) / 2 / (h * 2) * h
                    px = (cx, cy)
                yield np.concatenate(frames, axis=0), px
        cap.release()
        print(f"rendered+scanned match {m + 1}/{n_matches}", flush=True)


def soccernet_windows(data_dir: str, w: int, h: int):
    """Yield (3-frame stack, ball_pixel|None) from SoccerNet tracking sequences.

    Each sequence dir has ``img1/*.jpg`` frames and ``gt/gt.txt`` (MOT). The
    ball is the track whose class/role marks it as the ball; SoccerNet-Tracking
    labels the ball track, and it is by far the smallest box, which we use as a
    robust fallback identifier. Real ball GT — the domain a broadcast ball
    tracker actually needs.
    """
    import cv2
    import pandas as pd

    root = Path(data_dir)
    seqs = [d for d in sorted(root.iterdir()) if (d / "img1").exists()]
    if not seqs:
        raise SystemExit(f"no SoccerNet tracking sequences under {data_dir}")
    for seq in seqs:
        gt_path = seq / "gt" / "gt.txt"
        if not gt_path.exists():
            continue
        cols = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]
        gt = pd.read_csv(gt_path, header=None,
                         names=cols[: len(pd.read_csv(gt_path, header=None, nrows=1).columns)])
        gt["area"] = gt["w"] * gt["h"]
        # ball = the consistently-smallest-area track across the sequence
        med_area = gt.groupby("id")["area"].median()
        ball_id = int(med_area.idxmin())
        ball = gt[gt.id == ball_id].set_index("frame")
        imgs = sorted((seq / "img1").glob("*.jpg"))
        frames: list[np.ndarray] = []
        for ip in imgs:
            fnum = int(ip.stem)
            fr = cv2.imread(str(ip))
            H0, W0 = fr.shape[:2]
            small = cv2.resize(fr, (w, h))[:, :, ::-1].astype(np.float32) / 255.0
            frames.append(small.transpose(2, 0, 1))
            if len(frames) > 3:
                frames.pop(0)
            if len(frames) == 3:
                px = None
                if fnum in ball.index:
                    r = ball.loc[fnum]
                    px = (float(r.x + r.w / 2) / W0 * w, float(r.y + r.h / 2) / H0 * h)
                yield np.concatenate(frames, axis=0), px
        print(f"scanned sequence {seq.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "soccernet"], default="synthetic")
    ap.add_argument("--data-dir", default=None, help="SoccerNet tracking split dir")
    ap.add_argument("--matches", type=int, default=4)
    ap.add_argument("--half-minutes", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-windows", type=int, default=1600,
                    help="cap on stored training windows (keeps RAM bounded)")
    ap.add_argument("--out", default="weights/ball_tracknet.pt")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from pitchiq.perception.detection.tracknet import build_tracknet, gaussian_heatmap, TrackNetBall

    W, H = TrackNetBall.INPUT_SIZE
    hh, hw = H, W

    print("collecting training windows...")
    # store frames as uint8 (4x lighter than float32) + just the ball pixel;
    # heatmap targets are generated per-batch to keep RAM bounded
    stacks_u8: list[np.ndarray] = []
    ball_px: list[tuple | None] = []
    if args.source == "synthetic":
        gen = synthetic_windows(args.matches, args.half_minutes, W, H)
    else:
        if not args.data_dir:
            raise SystemExit("--source soccernet needs --data-dir <tracking split>")
        gen = soccernet_windows(args.data_dir, W, H)
    # reservoir-style subsample to keep at most --max-windows in RAM
    seen = 0
    keep_rng = np.random.default_rng(1)
    for stack, px in gen:
        seen += 1
        if len(stacks_u8) < args.max_windows:
            stacks_u8.append((stack * 255).astype(np.uint8))
            ball_px.append(px)
        else:
            j = int(keep_rng.integers(0, seen))
            if j < args.max_windows:
                stacks_u8[j] = (stack * 255).astype(np.uint8)
                ball_px[j] = px
    n = len(stacks_u8)
    n_ball = sum(p is not None for p in ball_px)
    print(f"{n} windows ({n_ball} with a visible ball)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_tracknet(3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)

    def make_batch(idx):
        xb = torch.from_numpy(
            np.stack([stacks_u8[i] for i in idx]).astype(np.float32) / 255.0).to(device)
        yb = torch.from_numpy(np.stack([
            gaussian_heatmap(hh, hw, *ball_px[i], sigma=3.0) if ball_px[i] is not None
            else np.zeros((hh, hw), np.float32) for i in idx])[:, None]).to(device)
        return xb, yb

    for ep in range(args.epochs):
        perm = rng.permutation(n)
        losses = []
        model.train()
        for i in range(0, n, args.batch):
            idx = perm[i: i + args.batch]
            xb, yb = make_batch(idx)
            pred = model(xb)
            # weighted BCE — the ball is a tiny positive region
            loss = F.binary_cross_entropy_with_logits(
                pred, yb, pos_weight=torch.tensor(200.0, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        print(f"epoch {ep + 1}/{args.epochs}: loss {np.mean(losses):.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    torch.jit.trace(model, torch.zeros(1, 9, H, W)).save(str(out))
    print(f"saved {out} — set detection.ball.tracknet_weights to use it.", flush=True)


if __name__ == "__main__":
    main()
