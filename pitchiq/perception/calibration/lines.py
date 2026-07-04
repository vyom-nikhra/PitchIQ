"""Pitch-line extraction from a broadcast frame.

Pipeline: grass mask → white top-hat response inside the field → probabilistic
Hough segments → merge collinear fragments → the long merged segments that
calibration matches against the pitch template.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pitchiq.core.geometry import angle_diff_deg, point_line_distance
from pitchiq.perception.detection.blob import grass_mask


@dataclass
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def angle(self) -> float:
        """Orientation in [0, 180) degrees."""
        return float(np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1)) % 180.0)

    @property
    def midpoint(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2])

    def endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([self.x1, self.y1]), np.array([self.x2, self.y2])

    def x_at_y(self, y: float) -> float:
        """x of the infinite line at height y (for near-vertical ordering)."""
        if abs(self.y2 - self.y1) < 1e-6:
            return (self.x1 + self.x2) / 2
        t = (y - self.y1) / (self.y2 - self.y1)
        return self.x1 + t * (self.x2 - self.x1)

    def y_at_x(self, x: float) -> float:
        if abs(self.x2 - self.x1) < 1e-6:
            return (self.y1 + self.y2) / 2
        t = (x - self.x1) / (self.x2 - self.x1)
        return self.y1 + t * (self.y2 - self.y1)


def white_line_mask(frame_bgr: np.ndarray, field: np.ndarray | None = None) -> np.ndarray:
    """Binary mask of white pitch markings inside the grass region.

    A morphological top-hat on the grayscale image responds to thin bright
    structures regardless of absolute brightness, which handles shadow bands
    better than a global threshold; an HSV low-saturation gate then rejects
    bright but coloured responses (kits, boards).
    """
    if field is None:
        field = grass_mask(frame_bgr)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, th = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    low_sat = cv2.inRange(hsv, (0, 0, 120), (180, 90, 255))
    mask = cv2.bitwise_and(th, low_sat)
    mask = cv2.bitwise_and(mask, field)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def detect_segments(mask: np.ndarray, min_len_frac: float = 0.05) -> list[Segment]:
    """Probabilistic Hough segments on the white-line mask."""
    h, w = mask.shape[:2]
    min_len = int(min_len_frac * max(h, w))
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 360,
        threshold=60,
        minLineLength=min_len,
        maxLineGap=int(0.02 * max(h, w)),
    )
    if lines is None:
        return []
    return [Segment(*map(float, row)) for row in np.asarray(lines).reshape(-1, 4)]


def merge_segments(
    segments: list[Segment], angle_tol: float = 3.0, dist_tol: float = 10.0
) -> list[Segment]:
    """Merge collinear Hough fragments into full marking lines.

    Two segments merge when their orientations agree within ``angle_tol``
    degrees and each midpoint lies within ``dist_tol`` pixels of the other's
    infinite line. The merged segment spans the extreme projections of all
    member endpoints — one physical pitch line ends up as one segment.
    """
    remaining = sorted(segments, key=lambda s: -s.length)
    merged: list[Segment] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        rest = []
        for s in remaining:
            if angle_diff_deg(s.angle, seed.angle) > angle_tol:
                rest.append(s)
                continue
            a, b = seed.endpoints()
            d1 = point_line_distance(s.midpoint[None], a, b)[0]
            a2, b2 = s.endpoints()
            d2 = point_line_distance(seed.midpoint[None], a2, b2)[0]
            if min(d1, d2) <= dist_tol:
                group.append(s)
            else:
                rest.append(s)
        remaining = rest
        pts = np.concatenate([np.stack(s.endpoints()) for s in group])
        a, b = seed.endpoints()
        direction = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        t = (pts - a) @ direction
        p_min = a + t.min() * direction
        p_max = a + t.max() * direction
        merged.append(Segment(p_min[0], p_min[1], p_max[0], p_max[1]))
    return merged


def extract_pitch_lines(
    frame_bgr: np.ndarray, min_len_frac: float = 0.05
) -> tuple[list[Segment], np.ndarray, np.ndarray]:
    """Full extraction: returns (merged segments, white mask, field mask)."""
    field = grass_mask(frame_bgr)
    mask = white_line_mask(frame_bgr, field)
    segs = merge_segments(detect_segments(mask, min_len_frac))
    return segs, mask, field


def split_line_families(segments: list[Segment]) -> tuple[list[Segment], list[Segment]]:
    """Split segments into the two dominant orientation families.

    In the standard broadcast view, pitch lines of constant world-y
    (touchlines, box side lines) appear near-horizontal, while constant-x
    lines (goal line, box fronts, halfway) appear steeper. We simply split at
    45° from horizontal, which is robust for the main camera; oblique replay
    angles are handled (or rejected) by hypothesis scoring downstream.

    Returns ``(h_family, v_family)`` — image-horizontal-ish, image-vertical-ish.
    """
    h_family = [s for s in segments if min(s.angle, 180 - s.angle) < 45.0]
    v_family = [s for s in segments if min(s.angle, 180 - s.angle) >= 45.0]
    return h_family, v_family
