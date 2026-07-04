"""Colour-blob fallback detector.

Finds non-grass blobs inside the grass region and classifies them by size and
shape: small bright round blob → ball, person-sized blob → player. This is
exact on PitchIQ's synthetic broadcast renders (players are drawn as kit-
coloured figures on green) and serves as the zero-dependency fallback so the
whole pipeline still runs end-to-end without torch. It is NOT expected to be
accurate on real broadcast footage — use the YOLO backend there.
"""

from __future__ import annotations

import cv2
import numpy as np

from pitchiq.core.types import Detection, EntityClass
from pitchiq.perception.detection.base import Detector

GRASS_LOWER = np.array([30, 40, 40], dtype=np.uint8)
GRASS_UPPER = np.array([95, 255, 255], dtype=np.uint8)


def grass_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Binary mask of the (largest connected) grass region."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GRASS_LOWER, GRASS_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == biggest, 255, 0).astype(np.uint8)


class ColorBlobDetector(Detector):
    name = "blob-fallback"

    def __init__(
        self,
        conf_floor: float = 0.3,
        min_person_area_frac: float = 4e-5,
        max_person_area_frac: float = 8e-3,
        max_ball_area_frac: float = 3.2e-4,
    ) -> None:
        self.conf_floor = conf_floor
        self.min_person_area_frac = min_person_area_frac
        self.max_person_area_frac = max_person_area_frac
        self.max_ball_area_frac = max_ball_area_frac

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w)
        field = grass_mask(frame_bgr)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        non_grass = cv2.bitwise_not(cv2.inRange(hsv, GRASS_LOWER, GRASS_UPPER))
        # ignore white pitch lines: thin structures removed by opening
        blobs = cv2.bitwise_and(non_grass, field)
        blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        blobs = cv2.morphologyEx(blobs, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        n, labels, stats, _ = cv2.connectedComponentsWithStats(blobs, connectivity=8)
        dets: list[Detection] = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            frac = area / frame_area
            if frac < self.min_person_area_frac * 0.3 or frac > self.max_person_area_frac:
                continue
            aspect = bh / max(bw, 1)
            x1, y1, x2, y2 = float(x), float(y), float(x + bw), float(y + bh)
            crop = frame_bgr[y : y + bh, x : x + bw]
            mean_val = float(crop.mean()) if crop.size else 0.0
            roundish = 0.6 <= aspect <= 1.7
            if frac <= self.max_ball_area_frac and roundish and mean_val > 150:
                # small bright round blob → ball
                dets.append(
                    Detection(
                        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                        conf=max(self.conf_floor, 0.5),
                        cls=EntityClass.BALL,
                    )
                )
            elif frac >= self.min_person_area_frac and aspect > 0.9:
                conf = min(0.95, 0.55 + 3.0 * np.sqrt(frac))
                dets.append(
                    Detection(
                        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                        conf=float(conf),
                        cls=EntityClass.PLAYER,
                    )
                )
        return dets
