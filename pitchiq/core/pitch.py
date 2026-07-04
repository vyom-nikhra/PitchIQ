"""Standard pitch model: dimensions, marking lines, semantic keypoints, zones.

Coordinate convention (used everywhere in PitchIQ):

    x in [0, length_m]  — along the touchlines, left goal line at x=0
    y in [0, width_m]   — along the goal lines,  one touchline at y=0

Units are metres on a FIFA-standard 105 x 68 pitch. Which team attacks
positive-x is stored in match metadata (``attack_direction``), and analytics
normalise with :func:`to_attacking_coords` when a team-relative view is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# FIFA standard marking dimensions (metres)
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_WIDTH = 18.32
PENALTY_SPOT_DIST = 11.0
CIRCLE_RADIUS = 9.15
GOAL_WIDTH = 7.32
CORNER_ARC_RADIUS = 1.0


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float
    # visible angular range in radians (for penalty arcs); full circle if None
    theta_range: tuple[float, float] | None = None


@dataclass
class Pitch:
    """A parametric pitch. All lines/keypoints derive from length/width."""

    length: float = 105.0
    width: float = 68.0

    #: name -> ((x1, y1), (x2, y2)) straight marking segments
    lines: dict[str, tuple[tuple[float, float], tuple[float, float]]] = field(init=False)
    #: name -> (x, y) semantic keypoints (~33 of them)
    keypoints: dict[str, tuple[float, float]] = field(init=False)
    circles: dict[str, Circle] = field(init=False)

    def __post_init__(self) -> None:
        L, W = self.length, self.width
        pa_y0 = (W - PENALTY_AREA_WIDTH) / 2  # 13.84
        pa_y1 = W - pa_y0                     # 54.16
        ga_y0 = (W - GOAL_AREA_WIDTH) / 2     # 24.84
        ga_y1 = W - ga_y0                     # 43.16
        cy = W / 2
        # y-offset where the penalty arc meets the penalty-area front line
        arc_dy = math.sqrt(CIRCLE_RADIUS**2 - (PENALTY_AREA_DEPTH - PENALTY_SPOT_DIST) ** 2)

        self.lines = {
            "touch_bottom": ((0, 0), (L, 0)),
            "touch_top": ((0, W), (L, W)),
            "goal_left": ((0, 0), (0, W)),
            "goal_right": ((L, 0), (L, W)),
            "halfway": ((L / 2, 0), (L / 2, W)),
            # left penalty area
            "pa_left_front": ((PENALTY_AREA_DEPTH, pa_y0), (PENALTY_AREA_DEPTH, pa_y1)),
            "pa_left_bottom": ((0, pa_y0), (PENALTY_AREA_DEPTH, pa_y0)),
            "pa_left_top": ((0, pa_y1), (PENALTY_AREA_DEPTH, pa_y1)),
            # left goal area
            "ga_left_front": ((GOAL_AREA_DEPTH, ga_y0), (GOAL_AREA_DEPTH, ga_y1)),
            "ga_left_bottom": ((0, ga_y0), (GOAL_AREA_DEPTH, ga_y0)),
            "ga_left_top": ((0, ga_y1), (GOAL_AREA_DEPTH, ga_y1)),
            # right penalty area
            "pa_right_front": ((L - PENALTY_AREA_DEPTH, pa_y0), (L - PENALTY_AREA_DEPTH, pa_y1)),
            "pa_right_bottom": ((L, pa_y0), (L - PENALTY_AREA_DEPTH, pa_y0)),
            "pa_right_top": ((L, pa_y1), (L - PENALTY_AREA_DEPTH, pa_y1)),
            # right goal area
            "ga_right_front": ((L - GOAL_AREA_DEPTH, ga_y0), (L - GOAL_AREA_DEPTH, ga_y1)),
            "ga_right_bottom": ((L, ga_y0), (L - GOAL_AREA_DEPTH, ga_y0)),
            "ga_right_top": ((L, ga_y1), (L - GOAL_AREA_DEPTH, ga_y1)),
        }

        self.keypoints = {
            "corner_bl": (0, 0),
            "corner_tl": (0, W),
            "corner_br": (L, 0),
            "corner_tr": (L, W),
            "halfway_bottom": (L / 2, 0),
            "halfway_top": (L / 2, W),
            "center_spot": (L / 2, cy),
            "circle_left": (L / 2 - CIRCLE_RADIUS, cy),
            "circle_right": (L / 2 + CIRCLE_RADIUS, cy),
            "circle_bottom": (L / 2, cy - CIRCLE_RADIUS),
            "circle_top": (L / 2, cy + CIRCLE_RADIUS),
            # left half
            "pa_left_front_bottom": (PENALTY_AREA_DEPTH, pa_y0),
            "pa_left_front_top": (PENALTY_AREA_DEPTH, pa_y1),
            "pa_left_goal_bottom": (0, pa_y0),
            "pa_left_goal_top": (0, pa_y1),
            "ga_left_front_bottom": (GOAL_AREA_DEPTH, ga_y0),
            "ga_left_front_top": (GOAL_AREA_DEPTH, ga_y1),
            "ga_left_goal_bottom": (0, ga_y0),
            "ga_left_goal_top": (0, ga_y1),
            "penalty_spot_left": (PENALTY_SPOT_DIST, cy),
            "arc_left_bottom": (PENALTY_AREA_DEPTH, cy - arc_dy),
            "arc_left_top": (PENALTY_AREA_DEPTH, cy + arc_dy),
            # right half (mirror)
            "pa_right_front_bottom": (L - PENALTY_AREA_DEPTH, pa_y0),
            "pa_right_front_top": (L - PENALTY_AREA_DEPTH, pa_y1),
            "pa_right_goal_bottom": (L, pa_y0),
            "pa_right_goal_top": (L, pa_y1),
            "ga_right_front_bottom": (L - GOAL_AREA_DEPTH, ga_y0),
            "ga_right_front_top": (L - GOAL_AREA_DEPTH, ga_y1),
            "ga_right_goal_bottom": (L, ga_y0),
            "ga_right_goal_top": (L, ga_y1),
            "penalty_spot_right": (L - PENALTY_SPOT_DIST, cy),
            "arc_right_bottom": (L - PENALTY_AREA_DEPTH, cy - arc_dy),
            "arc_right_top": (L - PENALTY_AREA_DEPTH, cy + arc_dy),
        }

        arc_half = math.acos((PENALTY_AREA_DEPTH - PENALTY_SPOT_DIST) / CIRCLE_RADIUS)
        self.circles = {
            "center": Circle(L / 2, cy, CIRCLE_RADIUS),
            # penalty arcs: only the part outside the box is drawn
            "arc_left": Circle(PENALTY_SPOT_DIST, cy, CIRCLE_RADIUS, (-arc_half, arc_half)),
            "arc_right": Circle(
                L - PENALTY_SPOT_DIST, cy, CIRCLE_RADIUS, (math.pi - arc_half, math.pi + arc_half)
            ),
        }

    # ------------------------------------------------------------------ zones

    def contains(self, x: np.ndarray, y: np.ndarray, margin: float = 2.0) -> np.ndarray:
        """Boolean mask of points on (or within ``margin`` metres of) the pitch."""
        x = np.asarray(x)
        y = np.asarray(y)
        return (
            (x >= -margin) & (x <= self.length + margin) & (y >= -margin) & (y <= self.width + margin)
        )

    def third_of(self, x: np.ndarray) -> np.ndarray:
        """0 = defensive, 1 = middle, 2 = final third (attacking positive-x)."""
        x = np.asarray(x, dtype=float)
        return np.digitize(x, [self.length / 3, 2 * self.length / 3])

    def in_penalty_area(self, x: np.ndarray, y: np.ndarray, side: str) -> np.ndarray:
        """Mask of points inside the ``left`` or ``right`` penalty area."""
        x = np.asarray(x)
        y = np.asarray(y)
        pa_y0 = (self.width - PENALTY_AREA_WIDTH) / 2
        pa_y1 = self.width - pa_y0
        if side == "left":
            return (x <= PENALTY_AREA_DEPTH) & (y >= pa_y0) & (y <= pa_y1) & (x >= 0)
        return (x >= self.length - PENALTY_AREA_DEPTH) & (y >= pa_y0) & (y <= pa_y1) & (x <= self.length)

    # -------------------------------------------------------------- sampling

    def sample_line_points(self, step: float = 1.0) -> np.ndarray:
        """Dense (N, 2) points along every straight marking — used to score a
        homography hypothesis against the detected white-line mask."""
        pts: list[np.ndarray] = []
        for (x1, y1), (x2, y2) in self.lines.values():
            n = max(2, int(math.hypot(x2 - x1, y2 - y1) / step))
            t = np.linspace(0, 1, n)
            pts.append(np.stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t], axis=1))
        for c in self.circles.values():
            t0, t1 = c.theta_range if c.theta_range else (0, 2 * math.pi)
            n = max(8, int(abs(t1 - t0) * c.r / step))
            th = np.linspace(t0, t1, n)
            pts.append(np.stack([c.cx + c.r * np.cos(th), c.cy + c.r * np.sin(th)], axis=1))
        return np.concatenate(pts, axis=0)

    def keypoint_array(self) -> tuple[list[str], np.ndarray]:
        """Keypoints as (names, (N, 2) array) in a stable order."""
        names = list(self.keypoints.keys())
        return names, np.array([self.keypoints[n] for n in names], dtype=np.float64)


def to_attacking_coords(
    xy: np.ndarray, attack_direction: int, length: float = 105.0, width: float = 68.0
) -> np.ndarray:
    """Normalise positions so the team always attacks positive-x.

    ``attack_direction`` is +1 if the team already attacks positive-x, -1 otherwise.
    Flips both axes for -1 so left/right footedness of moves is preserved.
    """
    xy = np.asarray(xy, dtype=float)
    if attack_direction >= 0:
        return xy.copy()
    out = xy.copy()
    out[..., 0] = length - out[..., 0]
    out[..., 1] = width - out[..., 1]
    return out
