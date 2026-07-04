"""Appearance embeddings for re-ID association.

Two backends behind :func:`create_embedder`:

* ``osnet`` — a torchscript OSNet re-ID model (path via config / env). OSNet is
  the standard person re-ID backbone; we load an exported ``.pt`` torchscript
  so no torchreid dependency is needed. If torch or the weights are missing we
  fall back automatically.
* ``colorhist`` — HSV histogram of the torso region. Crude but surprisingly
  effective for football (kit colour + skin/hair) and dependency-free.

Embeddings are L2-normalised so cosine similarity is a dot product.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from pitchiq.config import AppearanceConfig

log = logging.getLogger(__name__)


class ColorHistEmbedder:
    """48-dim HSV histogram (16 bins x 3 channels) over the torso crop."""

    name = "colorhist"
    dim = 48

    def embed(self, frame_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_box = y2 - y1
        # torso: middle 60% width, 15-55% height (avoids grass, shorts, feet)
        cx1 = x1 + int(0.2 * (x2 - x1))
        cx2 = x2 - int(0.2 * (x2 - x1))
        cy1 = y1 + int(0.15 * h_box)
        cy2 = y1 + int(0.55 * h_box)
        crop = frame_bgr[max(0, cy1) : max(0, cy2), max(0, cx1) : max(0, cx2)]
        if crop.size < 48:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        feats = []
        for ch, rng in ((0, 180), (1, 256), (2, 256)):
            hist = cv2.calcHist([hsv], [ch], None, [16], [0, rng]).ravel()
            feats.append(hist)
        v = np.concatenate(feats).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None


class OSNetEmbedder:
    """Torchscript OSNet; expects a 256x128 RGB person crop."""

    name = "osnet"

    def __init__(self, weights_path: str) -> None:
        import torch  # local import: torch optional

        self.torch = torch
        self.model = torch.jit.load(weights_path, map_location="cpu").eval()
        self.dim = 512

    def embed(self, frame_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size < 100:
            return None
        img = cv2.resize(crop, (128, 256))[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        with self.torch.no_grad():
            t = self.torch.from_numpy(img.transpose(2, 0, 1)[None].astype(np.float32))
            out = self.model(t).numpy().ravel()
        n = np.linalg.norm(out)
        return (out / n).astype(np.float32) if n > 0 else None


def create_embedder(cfg: AppearanceConfig):
    """Return an embedder per config, or None when appearance is disabled."""
    if not cfg.enabled:
        return None
    if cfg.backend in ("osnet", "auto"):
        weights = os.environ.get("PITCHIQ_OSNET_WEIGHTS", "weights/osnet_x0_25.torchscript.pt")
        if os.path.exists(weights):
            try:
                emb = OSNetEmbedder(weights)
                log.info("appearance embedder: OSNet (%s)", weights)
                return emb
            except Exception as exc:
                log.warning("OSNet load failed (%s); using colour histograms", exc)
        elif cfg.backend == "osnet":
            log.warning("OSNet weights not found at %s; using colour histograms", weights)
    return ColorHistEmbedder()
