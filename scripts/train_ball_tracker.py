"""Train the TrackNet-style heatmap ball tracker — properly.

Two data sources (mirrors the keypoint model's synthetic/real split):

* ``--source soccernet`` — real ball ground truth from SoccerNet-Tracking
  MOT sequences. What a real-broadcast ball tracker needs. NDA: local only.
* ``--source synthetic`` — render simulated matches; the renderer knows the
  exact ball pixel, so data is unlimited and licence-clean (synthetic domain).

Procedure (designed for low loss AND generalisation):
  sequence-level train/val split -> lazy disk-backed dataset with a resize
  cache -> augmentation -> CenterNet focal loss -> AdamW + warmup/cosine ->
  mixed precision -> per-epoch validation (detection rate, false positives,
  pixel error on held-out sequences) -> best checkpoint + early stopping.
Checkpoints and a JSON log land in ``weights/ball_tracknet_train/`` so an
interrupted run resumes with ``--resume``.

Usage:
    python scripts/train_ball_tracker.py --source soccernet \
        --data-dir data/soccernet/tracking/tracking/test --epochs 30
    python scripts/train_ball_tracker.py --source synthetic --matches 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from pitchiq.perception.detection.ball_dataset import (  # noqa: E402
    BallWindowDataset,
    focal_heatmap_loss,
    list_ball_windows,
)


# ------------------------------------------------------------- data sources
def soccernet_split(data_dir: Path, n_val: int, stride: int, val_stride: int,
                    limit_seqs: int = 0):
    """Sequence-level split: whole clips held out for validation, spread
    across the sorted list so val sees varied matches/conditions."""
    seqs = sorted(d for d in data_dir.iterdir() if (d / "img1").is_dir())
    if limit_seqs:
        seqs = seqs[:limit_seqs]
    if len(seqs) < 2:
        raise SystemExit(f"need >=2 sequences under {data_dir}, found {len(seqs)}")
    n_val = max(1, min(n_val, len(seqs) - 1))
    step = max(1, len(seqs) // n_val)
    val_names = {s.name for s in seqs[::step][:n_val]}
    train_w, val_w = [], []
    for s in seqs:
        if s.name in val_names:
            val_w += list_ball_windows(s, stride=val_stride)
        else:
            train_w += list_ball_windows(s, stride=stride)
        print(f"  scanned {s.name} ({'val' if s.name in val_names else 'train'})",
              flush=True)
    print(f"train windows: {len(train_w)} | val windows: {len(val_w)} "
          f"| val seqs: {sorted(val_names)}", flush=True)
    return train_w, val_w


def synthetic_split(n_matches: int, half_minutes: float):
    """Render matches, dump frames to jpgs, return (train, val) window lists
    in the same (paths, fraction) format — the last match is validation."""
    import cv2

    from pitchiq.config import load_config
    from pitchiq.core.pitch import Pitch
    from pitchiq.core.schema import BALL_ID
    from pitchiq.demo.render import render_match
    from pitchiq.demo.simulate import simulate_demo_match

    tmp = REPO / "data" / "jobs" / "_ball_train"
    per_match: list[list] = []
    for m in range(n_matches):
        cfg = load_config(overrides={"simulator": {"half_minutes": half_minutes,
                                                   "seed": 200 + m}})
        sim = simulate_demo_match(cfg.simulator)
        vid = tmp / f"m{m}.mp4"
        frames_dir = tmp / f"m{m}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        rw, rh = 1024, 576
        rr = render_match(sim.tracking, Pitch(), sim.meta.kit_colors, vid,
                          cfg.simulator.fps, width=rw, height=rh)
        boxes = rr.boxes[rr.boxes.entity_id == BALL_ID]
        ball = {int(r["frame"]): ((r["x1"] + r["x2"]) / 2 / rw,
                                  (r["y1"] + r["y2"]) / 2 / rh)
                for _, r in boxes.iterrows()}
        cap = cv2.VideoCapture(str(vid))
        paths, fidx = [], -1
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fidx += 1
            p = frames_dir / f"{fidx:06d}.jpg"
            if not p.exists():
                cv2.imwrite(str(p), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            paths.append(p)
        cap.release()
        wins = [([paths[i - 2], paths[i - 1], paths[i]], ball.get(i))
                for i in range(2, len(paths))]
        per_match.append(wins)
        print(f"  rendered match {m + 1}/{n_matches}: {len(wins)} windows", flush=True)
    train = [w for ws in per_match[:-1] for w in ws] or per_match[-1]
    val = per_match[-1]
    return train, val


# ---------------------------------------------------------------- validation
def run_validation(model, loader, device: str, thresh: float,
                   amp_dtype=None, amp_enabled: bool | None = None) -> dict:
    """Detection rate / false-positive rate / pixel error on held-out clips."""
    import torch

    if amp_dtype is None:
        amp_dtype = torch.float16
    if amp_enabled is None:
        amp_enabled = device == "cuda"
    model.eval()
    n_pos = n_neg = det = fp = 0
    errs: list[float] = []
    with torch.no_grad():
        for xb, _, has, px in loader:
            xb = xb.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=amp_enabled):
                prob = torch.sigmoid(model(xb))[:, 0]
            B, _, Ww = prob.shape
            flat = prob.reshape(B, -1).float()
            peak, arg = flat.max(dim=1)
            iy, ix = (arg // Ww).float(), (arg % Ww).float()
            for b in range(B):
                if float(has[b]) > 0.5:
                    n_pos += 1
                    if float(peak[b]) >= thresh:
                        det += 1
                        errs.append(float(np.hypot(float(ix[b]) - float(px[b, 0]),
                                                   float(iy[b]) - float(px[b, 1]))))
                else:
                    n_neg += 1
                    if float(peak[b]) >= thresh:
                        fp += 1
    det_rate = det / max(n_pos, 1)
    fp_rate = fp / max(n_neg, 1)
    med = float(np.median(errs)) if errs else float("nan")
    p90 = float(np.percentile(errs, 90)) if errs else float("nan")
    # composite used for best-model selection: find the ball, don't invent
    # one, and be close (error term saturates at 20 px so it can't dominate)
    score = det_rate - fp_rate - (min(med, 20.0) / 200.0 if errs else 0.1)
    return dict(det_rate=det_rate, fp_rate=fp_rate, med_err_px=med,
                p90_err_px=p90, n_pos=n_pos, n_neg=n_neg, score=score)


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["soccernet", "synthetic"], default="soccernet")
    ap.add_argument("--data-dir", default="data/soccernet/tracking/tracking/test")
    ap.add_argument("--matches", type=int, default=4, help="synthetic only")
    ap.add_argument("--half-minutes", type=float, default=0.5, help="synthetic only")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--stride", type=int, default=2,
                    help="train window stride (2 halves near-duplicate frames)")
    ap.add_argument("--val-stride", type=int, default=5)
    ap.add_argument("--val-seqs", type=int, default=6)
    ap.add_argument("--limit-seqs", type=int, default=0, help="smoke runs")
    ap.add_argument("--thresh", type=float, default=0.35)
    ap.add_argument("--patience", type=int, default=8,
                    help="early-stop after this many epochs without val improvement")
    ap.add_argument("--out", default="weights/ball_tracknet.pt")
    ap.add_argument("--run-dir", default="weights/ball_tracknet_train",
                    help="checkpoint/log dir; point at persistent storage "
                         "(e.g. /kaggle/working/...) so --resume survives a "
                         "session death")
    ap.add_argument("--fp32", action="store_true",
                    help="disable mixed precision entirely (escape hatch for "
                         "fp16-only GPUs if overflow guards ever fall short)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from pitchiq.perception.detection.tracknet import TrackNetBall, build_tracknet

    input_size = TrackNetBall.INPUT_SIZE  # (w, h)
    run_dir = REPO / args.run_dir  # pathlib: an absolute --run-dir wins
    run_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = run_dir / "last.ckpt"
    log_path = run_dir / "log.json"

    print("building window lists...", flush=True)
    if args.source == "soccernet":
        cache = REPO / "data" / "soccernet" / "cache_ball_512x288"
        train_w, val_w = soccernet_split(REPO / args.data_dir, args.val_seqs,
                                         args.stride, args.val_stride,
                                         args.limit_seqs)
    else:
        cache = None
        train_w, val_w = synthetic_split(args.matches, args.half_minutes)

    train_ds = BallWindowDataset(train_w, input_size, augment=True, cache_dir=cache)
    val_ds = BallWindowDataset(val_w, input_size, augment=False, cache_dir=cache)
    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=pin,
                          persistent_workers=args.workers > 0, drop_last=True)
    # validation loads in-process: a second persistent worker pool means
    # another N full torch processes (~0.5 GB each) resident for the whole
    # run — enough to OOM an 8 GB machine. Val is small and cache-backed.
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=0, pin_memory=pin)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_tracknet(3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_iters = max(1, len(train_dl) * args.epochs)
    warmup = min(500, total_iters // 20 + 1)

    def lr_lambda(it: int) -> float:  # linear warmup then cosine to 1% of lr
        if it < warmup:
            return (it + 1) / warmup
        t = (it - warmup) / max(1, total_iters - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    # bf16 has float32-like range, so activations cannot overflow the way
    # fp16's 65504 ceiling allows (observed: inf activations poisoned the
    # BatchNorm running stats even though the skip-guard protected the
    # weights). fp16 stays as the fallback for pre-Ampere GPUs (P100/T4),
    # protected by a logit clamp + the skip-guard + the epoch health check.
    # NOTE: gate on compute capability, not is_bf16_supported() — torch
    # reports bf16 as "supported" on a T4 but emulates it without tensor
    # cores (a Kaggle T4 epoch took 2.2 h in bf16 vs ~fp16's tens of minutes).
    amp_enabled = device == "cuda" and not args.fp32
    use_bf16 = amp_enabled and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print("mixed precision:",
          "bf16" if use_bf16 else ("fp16" if amp_enabled else "off (fp32)"),
          flush=True)
    scaler = torch.amp.GradScaler(enabled=amp_enabled and not use_bf16)

    start_ep, best_score, since_best, history = 0, -1e9, 0, []
    if args.resume and last_ckpt.exists():
        # our own checkpoint (holds the history dict, not just tensors), so
        # PyTorch 2.6's weights_only=True default must be opted out of
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep = ck["epoch"] + 1
        best_score = ck["best_score"]
        since_best = ck.get("since_best", 0)
        history = ck.get("history", [])
        print(f"resumed from epoch {start_ep} (best score {best_score:.3f})", flush=True)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(start_ep, args.epochs):
        model.train()
        t0, losses, skipped = time.time(), [], 0
        for xb, yb, _, _ in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=amp_enabled):
                # clamp turns an fp16 activation blow-up (inf logits) into a
                # saturated-but-finite prediction instead of a poisoned batch
                loss = focal_heatmap_loss(model(xb).clamp(-30.0, 30.0), yb)
            if not torch.isfinite(loss):
                # never let a bad batch reach the weights (one NaN backward
                # poisoned an entire run before this guard existed)
                opt.zero_grad(set_to_none=True)
                skipped += 1
                sched.step()
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
        if skipped:
            print(f"  WARNING: skipped {skipped} non-finite batches", flush=True)
        if not losses or not np.isfinite(np.mean(losses)):
            raise RuntimeError(
                "training diverged: epoch produced no finite losses — aborting "
                "instead of writing a poisoned checkpoint")
        # health check BEFORE checkpointing: BatchNorm running stats update on
        # every train-mode forward, so inf activations can poison the model's
        # buffers even when the skip-guard blocks the gradient step
        bad = [n for n, t in model.state_dict().items()
               if not torch.isfinite(t).all()]
        if bad:
            raise RuntimeError(
                f"model poisoned (non-finite tensors: {bad[:4]}...) — aborting; "
                f"restart with --resume from the last healthy checkpoint")
        metrics = run_validation(model, val_dl, device, args.thresh,
                                 amp_dtype, amp_enabled)
        history.append(dict(epoch=ep, loss=float(np.mean(losses)), **metrics))
        print(f"epoch {ep + 1}/{args.epochs}: loss {np.mean(losses):.4f} | "
              f"val det {metrics['det_rate']:.2%} fp {metrics['fp_rate']:.2%} "
              f"med {metrics['med_err_px']:.1f}px p90 {metrics['p90_err_px']:.1f}px "
              f"score {metrics['score']:.3f} | {time.time() - t0:.0f}s", flush=True)

        if metrics["score"] > best_score:
            best_score, since_best = metrics["score"], 0
            model.eval().cpu()
            torch.jit.trace(model, torch.zeros(1, 9, input_size[1], input_size[0])
                            ).save(str(out))
            model.to(device)
            print(f"  new best -> {out}", flush=True)
        else:
            since_best += 1
        # checkpoint AFTER the best-score update: saving before it stored a
        # stale best, so a resumed run treated its first epoch as a new best
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), epoch=ep,
                        best_score=best_score, since_best=since_best,
                        history=history), last_ckpt)
        log_path.write_text(json.dumps(history, indent=1))
        if since_best >= args.patience:
            print(f"early stop: no val improvement in {args.patience} epochs",
                  flush=True)
            break

    print(f"done. best val score {best_score:.3f}; weights at {out}", flush=True)


if __name__ == "__main__":
    main()
