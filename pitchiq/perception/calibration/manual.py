"""Manual calibration: solve H from user-clicked point correspondences.

The web app exposes this as the escape hatch when automatic calibration fails
on a difficult clip: the user clicks ≥4 identifiable pitch points on a frame
(picked from the named keypoint list) and we solve directly.
"""

from __future__ import annotations

import cv2
import numpy as np

from pitchiq.core.geometry import reprojection_error
from pitchiq.core.pitch import Pitch


def homography_from_correspondences(
    image_points: np.ndarray, pitch_points: np.ndarray
) -> tuple[np.ndarray, float]:
    """Solve pixel→pitch H from matched points. Returns (H, rms_error_m)."""
    image_points = np.asarray(image_points, dtype=np.float64)
    pitch_points = np.asarray(pitch_points, dtype=np.float64)
    if len(image_points) < 4:
        raise ValueError("need at least 4 correspondences")
    method = cv2.RANSAC if len(image_points) > 4 else 0
    H, _ = cv2.findHomography(image_points, pitch_points, method, 3.0)
    if H is None:
        raise ValueError("homography estimation failed (degenerate points?)")
    return H, reprojection_error(H, image_points, pitch_points)


def homography_from_named_keypoints(
    clicks: dict[str, tuple[float, float]], pitch: Pitch
) -> tuple[np.ndarray, float]:
    """Solve from {keypoint_name: (px, py)} clicks using the pitch's keypoint table."""
    img, world = [], []
    for name, xy in clicks.items():
        if name not in pitch.keypoints:
            raise KeyError(f"unknown pitch keypoint '{name}'")
        img.append(xy)
        world.append(pitch.keypoints[name])
    return homography_from_correspondences(np.array(img), np.array(world))
