"""TrackNet-style heatmap ball tracker.

The ball is the least reliable perception target — small, fast, motion-blurred,
occluded, sometimes invisible. Box detectors (YOLO) treat each frame
independently and miss exactly these cases. TrackNet (Huang et al. 2019) and
successors instead regress a **ball probability heatmap from several
consecutive frames**, so the network learns the *trajectory* and can localise
the ball through blur/occlusion where a single-frame detector fails.

This implementation: 3 consecutive RGB frames (9 input channels) → one
full-resolution heatmap for the latest frame. Ball position is the sub-pixel
centroid around the heatmap peak, gated by peak confidence. A small
encoder–decoder keeps inference cheap.

Training data: PitchIQ's synthetic renderer knows the exact ball pixel every
frame (licence-clean, unlimited), and SoccerNet tracking provides real ball
ground truth — both wired in ``scripts/train_ball_tracker.py``. Like the
pitch-keypoint model, a model trained on synthetic renders is for the
synthetic domain; real broadcast needs real training data. Without weights the
pipeline falls back to the YOLO+Kalman ball selector.

Documented limitation (unchanged): an airborne ball still projects incorrectly
through the ground-plane homography — inherent to single-camera geometry.
"""

from __future__ import annotations

import logging
from collections import deque

import cv2
import numpy as np

log = logging.getLogger(__name__)


def build_tracknet(in_frames: int = 3):
    """Small TrackNet-style encoder–decoder. Input (B, 3*in_frames, H, W) →
    (B, 1, H, W) logits."""
    import torch.nn as nn

    def cbr(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
        )

    class TrackNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = 3 * in_frames
            self.e1 = nn.Sequential(cbr(c, 32), cbr(32, 32))
            self.e2 = nn.Sequential(cbr(32, 64), cbr(64, 64))
            self.e3 = nn.Sequential(cbr(64, 128), cbr(128, 128))
            self.pool = nn.MaxPool2d(2)
            self.bott = nn.Sequential(cbr(128, 256), cbr(256, 256))
            self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
            self.d3 = nn.Sequential(cbr(256, 128), cbr(128, 128))
            self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
            self.d2 = nn.Sequential(cbr(128, 64), cbr(64, 64))
            self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
            self.d1 = nn.Sequential(cbr(64, 32), cbr(32, 32))
            self.head = nn.Conv2d(32, 1, 1)

        def forward(self, x):
            import torch

            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            b = self.bott(self.pool(e3))
            d3 = self.d3(torch.cat([self.up3(b), e3], 1))
            d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
            return self.head(d1)

    return TrackNet()


class TrackNetBall:
    """Inference wrapper: buffers frames, returns the ball pixel per frame."""

    INPUT_SIZE = (512, 288)  # (w, h) — model resolution; frames resized in/out

    def __init__(self, weights_path: str, in_frames: int = 3,
                 device: str = "auto", peak_threshold: float = 0.5) -> None:
        import torch

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.in_frames = in_frames
        self.peak_threshold = peak_threshold
        try:
            self.model = torch.jit.load(weights_path, map_location=device).eval()
        except (RuntimeError, ValueError):
            self.model = build_tracknet(in_frames)
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
            self.model.eval().to(device)
        self._buf: deque[np.ndarray] = deque(maxlen=in_frames)
        log.info("TrackNet ball tracker loaded: %s (device=%s)", weights_path, device)

    def reset(self) -> None:
        self._buf.clear()

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame_bgr, self.INPUT_SIZE)[:, :, ::-1].astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)  # (3, H, W)

    def detect(self, frame_bgr: np.ndarray) -> tuple[float, float, float] | None:
        """Push a frame; return (x_px, y_px, confidence) at full resolution, or
        None while the buffer is filling or when no confident ball is found."""
        h0, w0 = frame_bgr.shape[:2]
        self._buf.append(self._preprocess(frame_bgr))
        if len(self._buf) < self.in_frames:
            return None
        stack = np.concatenate(list(self._buf), axis=0)[None]  # (1, 3*in, H, W)
        t = self.torch.from_numpy(stack.copy()).to(self.device)
        with self.torch.no_grad():
            heat = self.torch.sigmoid(self.model(t))[0, 0].cpu().numpy()
        return self._peak(heat, w0, h0)

    def _peak(self, heat: np.ndarray, w0: int, h0: int) -> tuple[float, float, float] | None:
        peak = float(heat.max())
        if peak < self.peak_threshold:
            return None
        hh, hw = heat.shape
        iy, ix = np.unravel_index(int(heat.argmax()), heat.shape)
        # sub-pixel: intensity-weighted centroid in a small window around the peak
        y0, y1 = max(0, iy - 3), min(hh, iy + 4)
        x0, x1 = max(0, ix - 3), min(hw, ix + 4)
        win = heat[y0:y1, x0:x1]
        ys, xs = np.mgrid[y0:y1, x0:x1]
        s = win.sum() + 1e-9
        cy = float((ys * win).sum() / s)
        cx = float((xs * win).sum() / s)
        return (cx / hw * w0, cy / hh * h0, peak)


def gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float = 3.0) -> np.ndarray:
    """Render a Gaussian ball-location target heatmap (training label)."""
    ys, xs = np.mgrid[0:h, 0:w]
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)).astype(np.float32)
