"""Learned crop embedders for team assignment.

Colour-histogram signatures collapse when two kits share a tone at low
resolution (e.g. white vs sky-blue on 576p broadcast). A learned image
embedding of the player crop separates them because it encodes texture, kit
pattern, crest and subtle colour jointly in a high-dimensional space where
K-Means finds the two-team structure that 11-dim colour statistics cannot.

Tiered by availability (all behind :func:`create_team_embedder`):

* ``siglip``    — SigLIP crop embeddings (the Roboflow-`sports` approach);
  best separation, needs ``transformers`` + a ~370 MB model.
* ``cnn``       — a torchvision ImageNet backbone (MobileNetV3-Small by
  default) global-pooled features; no extra install, ~10 MB weights.
* ``none``      — signals the caller to fall back to colour histograms.

Every embedder returns an L2-normalised vector so cosine == dot product.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


class CNNEmbedder:
    """Global-pooled features from a small pretrained torchvision backbone."""

    name = "cnn"

    def __init__(self, arch: str = "mobilenet_v3_small", device: str = "auto") -> None:
        import torch
        import torchvision

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        weights = "DEFAULT"
        model = getattr(torchvision.models, arch)(weights=weights)
        # strip the classifier: keep the conv feature extractor + global pool
        if hasattr(model, "classifier"):
            model.classifier = torch.nn.Identity()
        elif hasattr(model, "fc"):
            model.fc = torch.nn.Identity()
        self.model = model.eval().to(device)
        self.dim = None  # discovered on first embed
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size < 300:
            return None
        img = cv2.resize(crop_bgr, (96, 96))[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        t = self.torch.from_numpy(img.transpose(2, 0, 1)[None].copy()).to(self.device)
        with self.torch.no_grad():
            v = self.model(t).reshape(-1).cpu().numpy()
        n = np.linalg.norm(v)
        if n < 1e-8:
            return None
        self.dim = v.shape[0]
        return (v / n).astype(np.float32)


class SigLIPEmbedder:
    """SigLIP vision-tower crop embeddings (needs ``transformers``)."""

    name = "siglip"

    def __init__(self, model_id: str = "google/siglip-base-patch16-224",
                 device: str = "auto") -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval().to(device)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size < 300:
            return None
        rgb = crop_bgr[:, :, ::-1]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            feat = self.model.get_image_features(**inputs).reshape(-1).cpu().numpy()
        n = np.linalg.norm(feat)
        return (feat / n).astype(np.float32) if n > 1e-8 else None


def create_team_embedder(backend: str = "auto", device: str = "auto"):
    """Return a crop embedder, or ``None`` to use the colour fallback.

    ``backend``: 'auto' (siglip→cnn→none) | 'siglip' | 'cnn' | 'none'.
    """
    if backend == "none":
        return None
    if backend in ("siglip", "auto"):
        try:
            emb = SigLIPEmbedder(device=device)
            log.info("team embedder: SigLIP")
            return emb
        except Exception as exc:
            if backend == "siglip":
                log.warning("SigLIP requested but unavailable (%s); using CNN", exc)
    if backend in ("cnn", "auto", "siglip"):
        try:
            emb = CNNEmbedder(device=device)
            log.info("team embedder: torchvision CNN (%s)", emb.model.__class__.__name__)
            return emb
        except Exception as exc:
            log.warning("CNN embedder unavailable (%s); using colour histograms", exc)
    return None
