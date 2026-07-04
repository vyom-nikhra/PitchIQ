"""Fundamental value types used across perception and analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class EntityClass(str, Enum):
    """What a detected/tracked entity is."""

    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    BALL = "ball"


class Team(str, Enum):
    """Team label. ``NONE`` covers referees and the ball."""

    HOME = "home"
    AWAY = "away"
    NONE = "none"


PERSON_CLASSES = (EntityClass.PLAYER, EntityClass.GOALKEEPER, EntityClass.REFEREE)


@dataclass
class Detection:
    """A single-frame detection in pixel space.

    ``bbox`` is ``[x1, y1, x2, y2]`` in pixels. ``feature`` optionally carries an
    appearance embedding used by the tracker's re-ID association.
    """

    bbox: np.ndarray
    conf: float
    cls: EntityClass
    feature: Optional[np.ndarray] = None

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-center of the box — the point that projects onto the pitch plane."""
        x1, y1, x2, y2 = self.bbox
        return (float(x1 + x2) / 2.0, float(y2))

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


@dataclass
class TrackedEntity:
    """A tracker output for one entity on one frame."""

    track_id: int
    bbox: np.ndarray
    conf: float
    cls: EntityClass
    team: Team = Team.NONE
    jersey_no: Optional[int] = None

    @property
    def foot_point(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (float(x1 + x2) / 2.0, float(y2))


@dataclass
class FrameResult:
    """Everything perception produced for one frame."""

    frame_idx: int
    timestamp: float
    entities: list[TrackedEntity] = field(default_factory=list)
    homography: Optional[np.ndarray] = None  # 3x3 pixel -> pitch metres
    reproj_error_px: float = float("nan")
    calib_method: str = "none"
    is_scene_cut: bool = False


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two arrays of ``[x1, y1, x2, y2]`` boxes.

    Returns an ``(len(a), len(b))`` matrix. Vectorised; safe for empty inputs.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float64)[:, None, :]  # (A,1,4)
    b = np.asarray(boxes_b, dtype=np.float64)[None, :, :]  # (1,B,4)
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)
