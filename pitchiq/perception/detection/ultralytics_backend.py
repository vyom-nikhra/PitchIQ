"""Ultralytics-backed detectors: YOLOv11 and RT-DETR behind one interface.

With fine-tuned football weights (``detection.weights``) the model emits the
four PitchIQ classes directly. Without them we fall back to COCO pretraining:
``person`` → player and ``sports ball`` → ball. In COCO-fallback mode
goalkeepers/referees arrive labelled ``player`` and are recovered later by the
team-assignment stage (colour outliers + positional priors) — documented
limitation, fixed properly by running ``scripts/train_detector.py`` on the
Roboflow football dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pitchiq.config import DetectionConfig
from pitchiq.core.types import Detection, EntityClass
from pitchiq.perception.detection.base import Detector

log = logging.getLogger(__name__)


def download_weights(url: str, dest: str | Path, timeout: int = 60) -> None:
    """Stream ``url`` to ``dest`` (via a .part temp file so a failed download
    never leaves a corrupt weights file behind). Raises OSError on failure."""
    import urllib.request

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading detector weights: %s -> %s", url, dest)
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(part, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    part.replace(dest)
    log.info("detector weights ready (%.1f MB)", dest.stat().st_size / 1e6)


def resolve_weights(cfg: DetectionConfig) -> str | None:
    """The fine-tuned weights path if usable, else None (COCO fallback).

    A configured-but-absent file tries ``weights_url`` first; if that fails
    (offline, URL gone) the detector degrades to the COCO base model with a
    warning — the documented no-weights behaviour — instead of erroring all
    the way down to the blob detector.
    """
    if cfg.weights is None:
        return None
    if Path(cfg.weights).exists():
        return cfg.weights
    if cfg.weights_url:
        try:
            download_weights(cfg.weights_url, cfg.weights)
            return cfg.weights
        except OSError as exc:
            log.warning("detector weights download failed (%s)", exc)
    log.warning("configured detector weights missing: %s — using the COCO base "
                "model instead", cfg.weights)
    return None

# model class-name -> PitchIQ entity class (matched case-insensitively, substring)
NAME_MAP = {
    "ball": EntityClass.BALL,
    "sports ball": EntityClass.BALL,
    "goalkeeper": EntityClass.GOALKEEPER,
    "referee": EntityClass.REFEREE,
    "player": EntityClass.PLAYER,
    "person": EntityClass.PLAYER,
}


def map_class_name(name: str) -> EntityClass | None:
    low = name.lower()
    for key, ent in NAME_MAP.items():
        if key in low:
            return ent
    return None


class UltralyticsDetector(Detector):
    def __init__(self, cfg: DetectionConfig, arch: str = "yolo") -> None:
        from ultralytics import RTDETR, YOLO  # heavy import kept local

        self.cfg = cfg
        weights = resolve_weights(cfg)
        if weights is None:
            if not cfg.coco_fallback:
                raise RuntimeError("no football weights configured and coco_fallback disabled")
            weights = cfg.model if arch == "yolo" else "rtdetr-l.pt"
            log.warning(
                "No fine-tuned football weights; using COCO-pretrained %s. "
                "GK/referee classes will be recovered heuristically downstream.",
                weights,
            )
        self.model = YOLO(weights) if arch == "yolo" else RTDETR(weights)
        self.arch = arch
        self.name = f"{arch}:{weights}"
        self.device = None if cfg.device == "auto" else cfg.device
        self.class_map: dict[int, EntityClass] = {}
        names = getattr(self.model, "names", {}) or {}
        for idx, cname in (names.items() if isinstance(names, dict) else enumerate(names)):
            ent = map_class_name(str(cname))
            if ent is not None:
                self.class_map[int(idx)] = ent
        if not self.class_map:
            raise RuntimeError(f"model {weights} has no football-mappable classes: {names}")

    def _predict(self, imgs, conf: float | None = None):
        return self.model.predict(
            imgs,
            conf=conf or self.cfg.conf_threshold,
            imgsz=self.cfg.imgsz,
            device=self.device,
            classes=sorted(self.class_map),
            verbose=False,
        )

    def _to_detections(self, result) -> list[Detection]:
        dets: list[Detection] = []
        boxes = result.boxes
        if boxes is None:
            return dets
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)
        for bb, cf, cl in zip(xyxy, confs, clses):
            ent = self.class_map.get(int(cl))
            if ent is None:
                continue
            dets.append(Detection(bbox=bb.astype(np.float32), conf=float(cf), cls=ent))
        return dets

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        return self._to_detections(self._predict(frame_bgr)[0])

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        return [self._to_detections(r) for r in self._predict(frames)]

    def detect_roi(self, crop_bgr: np.ndarray, conf: float) -> list[Detection]:
        """High-recall pass on a small crop (used by the ball ROI strategy)."""
        results = self.model.predict(
            crop_bgr,
            conf=conf,
            imgsz=max(320, min(960, max(crop_bgr.shape[:2]))),
            device=self.device,
            classes=sorted(self.class_map),
            verbose=False,
        )
        return self._to_detections(results[0])
