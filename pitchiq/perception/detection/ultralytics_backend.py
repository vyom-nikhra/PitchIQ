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

import numpy as np

from pitchiq.config import DetectionConfig
from pitchiq.core.types import Detection, EntityClass
from pitchiq.perception.detection.base import Detector

log = logging.getLogger(__name__)

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
        weights = cfg.weights
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
