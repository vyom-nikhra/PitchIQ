"""Detector interface + factory.

Three interchangeable backends behind one interface:

* ``yolo``   — Ultralytics YOLOv11 (fine-tuned football weights, or COCO fallback)
* ``rtdetr`` — Ultralytics RT-DETR behind the same interface (benchmarkable vs YOLO)
* ``blob``   — dependency-free colour-blob detector; correct on PitchIQ's synthetic
  broadcast renders and used as the graceful fallback when torch is unavailable

``backend: auto`` tries yolo → blob and logs what it picked.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from pitchiq.config import DetectionConfig
from pitchiq.core.types import Detection

log = logging.getLogger(__name__)


class Detector(ABC):
    """Per-frame object detector returning :class:`Detection` lists."""

    name: str = "base"

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect entities on a single BGR frame."""

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Default batch = loop; GPU backends override with true batching."""
        return [self.detect(f) for f in frames]


def create_detector(cfg: DetectionConfig) -> Detector:
    """Build the configured detector, falling back gracefully.

    Fallback chain for ``auto``: ultralytics YOLO (fine-tuned weights if given,
    else COCO-pretrained with person/sports-ball mapping) → colour-blob detector.
    The choice is logged; the pipeline also records it in ``meta.json`` so every
    result is traceable to the detector that produced it.
    """
    backend = cfg.backend
    if backend in ("yolo", "rtdetr"):
        from pitchiq.perception.detection.ultralytics_backend import UltralyticsDetector

        return UltralyticsDetector(cfg, arch=backend)
    if backend == "blob":
        from pitchiq.perception.detection.blob import ColorBlobDetector

        return ColorBlobDetector(conf_floor=cfg.conf_threshold)

    # auto
    try:
        from pitchiq.perception.detection.ultralytics_backend import UltralyticsDetector

        det = UltralyticsDetector(cfg, arch="yolo")
        log.info("detector auto-selected: %s", det.name)
        return det
    except Exception as exc:
        log.warning(
            "ultralytics backend unavailable (%s); using colour-blob fallback detector. "
            "Install extras `pitchiq[cv]` for real broadcast footage.",
            exc,
        )
        from pitchiq.perception.detection.blob import ColorBlobDetector

        return ColorBlobDetector(conf_floor=cfg.conf_threshold)
