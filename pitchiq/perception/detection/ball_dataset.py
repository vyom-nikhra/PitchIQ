"""Dataset + loss + augmentation for training the TrackNet ball tracker.

Lives in the library (not the script) so the local trainer, the tests and a
Kaggle notebook share one tested pipeline.

Design goals — *low loss AND generalisation*:

* **Lazy, disk-backed** windows: frames are read (and resized) per item, so
  training scales to arbitrarily many sequences without holding them in RAM.
  An optional cache directory stores the resized frames, so only the first
  epoch pays the full-resolution JPEG decode cost.
* **Sequence-level split** (done by the trainer): validation sequences are
  entire held-out clips, so val metrics measure generalisation to unseen
  matches — a frame-level split would leak near-duplicate frames.
* **Augmentation** (flip, photometric jitter, small affine, random occlusion)
  applied consistently to the 3 frames and the ball location.
* **CenterNet penalty-reduced focal loss** on a Gaussian heatmap. The peak
  pixel is stamped to exactly 1.0 — the loss treats only ``target == 1`` as a
  positive, so without the stamp there would be *no* positives and the model
  would collapse to all-zeros.

Ball locations are carried as **fractions of frame size** (fx, fy in [0,1]),
so windows are resolution-independent and cached resizes stay valid.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------- window lists
def list_ball_windows(seq: Path, stride: int = 1) -> list[tuple[list[Path], tuple | None]]:
    """All 3-frame windows of a SoccerNet-Tracking sequence with the ball
    location (fractional, or None when not annotated) for the latest frame.

    The ball track is identified as the id with the smallest median box area —
    SoccerNet MOT ground truth has no reliable class column, and the ball is
    an order of magnitude smaller than any person box.

    Parsed with numpy, not pandas: gt.txt is plain numeric CSV, and pandas'
    pyarrow string backend can hard-crash (access violation) when torch is
    loaded in the same process on Windows.
    """
    gt_path = seq / "gt" / "gt.txt"
    imgs = sorted((seq / "img1").glob("*.jpg"))
    if not gt_path.exists() or len(imgs) < 3:
        return []
    first = cv2.imread(str(imgs[0]))
    if first is None:
        return []
    h0, w0 = first.shape[:2]
    arr = np.loadtxt(gt_path, delimiter=",", ndmin=2)  # frame,id,x,y,w,h,...
    ids = arr[:, 1].astype(int)
    areas = arr[:, 4] * arr[:, 5]
    ball_id = min(np.unique(ids), key=lambda i: float(np.median(areas[ids == i])))
    ball = {int(r[0]): (float(r[2] + r[4] / 2) / w0, float(r[3] + r[5] / 2) / h0)
            for r in arr[ids == ball_id]}
    out: list[tuple[list[Path], tuple | None]] = []
    for i in range(2, len(imgs), stride):
        out.append(([imgs[i - 2], imgs[i - 1], imgs[i]], ball.get(int(imgs[i].stem))))
    return out


# ------------------------------------------------------------------- targets
def gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float = 3.0) -> np.ndarray:
    """Gaussian target with an exact 1.0 stamped at the (rounded) peak pixel —
    required by the focal loss's ``target == 1`` positive mask."""
    ys, xs = np.mgrid[0:h, 0:w]
    heat = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)).astype(np.float32)
    iy = int(np.clip(round(cy), 0, h - 1))
    ix = int(np.clip(round(cx), 0, w - 1))
    heat[iy, ix] = 1.0
    return heat


# -------------------------------------------------------------- augmentation
def _augment(frames: list[np.ndarray], px, w: int, h: int, rng):
    """Consistent augmentation of a 3-frame window + ball pixel (w×h space)."""
    if rng.random() < 0.5:  # horizontal flip
        frames = [np.ascontiguousarray(f[:, ::-1]) for f in frames]
        if px is not None:
            px = (w - 1 - px[0], px[1])
    if rng.random() < 0.8:  # photometric jitter, shared across the window
        gain = 1.0 + rng.uniform(-0.25, 0.25)
        bias = rng.uniform(-0.08, 0.08)
        frames = [np.clip(f * gain + bias, 0.0, 1.0) for f in frames]
    if rng.random() < 0.5:  # small shared affine (translate + scale)
        tx, ty = rng.uniform(-0.06, 0.06) * w, rng.uniform(-0.06, 0.06) * h
        s = 1.0 + rng.uniform(-0.1, 0.1)
        M = np.array([[s, 0, tx], [0, s, ty]], np.float32)
        frames = [cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                  for f in frames]
        if px is not None:
            px = (s * px[0] + tx, s * px[1] + ty)
            if not (0 <= px[0] < w and 0 <= px[1] < h):
                px = None  # ball augmented out of frame -> negative sample
    if rng.random() < 0.3:  # occlusion patch on the latest frame
        ow, oh = int(rng.uniform(0.05, 0.15) * w), int(rng.uniform(0.05, 0.15) * h)
        ox, oy = int(rng.uniform(0, w - ow)), int(rng.uniform(0, h - oh))
        frames[-1][oy:oy + oh, ox:ox + ow] = rng.random()
    return frames, px


# ----------------------------------------------------------------- dataset
class BallWindowDataset:
    """Torch-compatible dataset over 3-frame windows, read lazily from disk.

    ``windows``: list of ([path0, path1, path2], (fx, fy) | None).
    ``cache_dir``: optional; stores frames pre-resized to ``input_size`` so
    epochs after the first skip the full-resolution decode.
    """

    def __init__(self, windows, input_size=(512, 288), sigma: float = 3.0,
                 augment: bool = False, cache_dir: Path | None = None):
        self.windows = windows
        self.w, self.h = input_size
        self.sigma = sigma
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.windows)

    def _load_resized(self, p: Path) -> np.ndarray:
        if self.cache_dir is not None:
            key = hashlib.md5(str(p).encode()).hexdigest()
            cp = self.cache_dir / f"{key}_{self.w}x{self.h}.jpg"
            if cp.exists():
                img = cv2.imread(str(cp))
                if img is not None:
                    return img
            img = cv2.resize(cv2.imread(str(p)), (self.w, self.h))
            cv2.imwrite(str(cp), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            return img
        return cv2.resize(cv2.imread(str(p)), (self.w, self.h))

    def __getitem__(self, i: int):
        import torch

        paths, frac = self.windows[i]
        frames = [self._load_resized(p)[:, :, ::-1].astype(np.float32) / 255.0
                  for p in paths]
        px = None if frac is None else (frac[0] * self.w, frac[1] * self.h)
        if self.augment:
            # fresh entropy per call: safe under DataLoader workers, varied
            # across epochs (a fixed per-item seed would repeat augmentations)
            frames, px = _augment(frames, px, self.w, self.h, np.random.default_rng())
        stack = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
        if px is None:
            heat = np.zeros((self.h, self.w), np.float32)
            has = 0.0
            px = (-1.0, -1.0)
        else:
            heat = gaussian_heatmap(self.h, self.w, px[0], px[1], self.sigma)
            has = 1.0
        return (torch.from_numpy(np.ascontiguousarray(stack)),
                torch.from_numpy(heat)[None],
                torch.tensor(has, dtype=torch.float32),
                torch.tensor(px, dtype=torch.float32))


# -------------------------------------------------------------------- loss
def focal_heatmap_loss(pred_logits, target, alpha: float = 2.0, beta: float = 4.0):
    """CenterNet penalty-reduced focal loss on Gaussian heatmaps.

    ``pred_logits``: (B,1,H,W) raw scores; ``target``: (B,1,H,W) in [0,1] with
    an exact 1.0 at each ball peak. Peak pixels are positives; everything else
    is a soft negative down-weighted by ``(1 - target)^beta`` so the ring
    around the peak isn't punished as if it were background.

    Computed in float32 via ``logsigmoid`` regardless of autocast: in fp16 a
    confident sigmoid saturates to exactly 1.0 (a ``1 - 1e-6`` clamp is below
    fp16 resolution), so ``log(1 - pred)`` hits -inf and one such batch
    permanently poisons the weights with NaNs — observed at epoch 3 of the
    first real run. ``logsigmoid(-x) == log(1 - sigmoid(x))`` exactly, with no
    saturating subtraction.
    """
    import torch
    import torch.nn.functional as F

    x = pred_logits.float()
    target = target.float()
    pred = torch.sigmoid(x)
    log_p = F.logsigmoid(x)       # log(sigmoid(x)), stable for any x
    log_1mp = F.logsigmoid(-x)    # log(1 - sigmoid(x)), stable for any x
    pos = (target == 1.0).float()
    neg = 1.0 - pos
    pos_loss = -((1 - pred) ** alpha) * log_p * pos
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * log_1mp * neg
    n_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos
