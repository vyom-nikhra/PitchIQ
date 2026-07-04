"""Kalman filter for bounding-box tracks (ByteTrack/SORT formulation).

State: (cx, cy, a, h, vcx, vcy, va, vh) with a = aspect ratio w/h.
Constant-velocity motion; noise scaled with box height as in the reference
implementations (std_weight_position=1/20, std_weight_velocity=1/160).
"""

from __future__ import annotations

import numpy as np


class KalmanBoxFilter:
    ndim = 4

    def __init__(self) -> None:
        dt = 1.0
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, i + 4] = dt
        self._H = np.eye(4, 8)
        self._std_pos = 1.0 / 20
        self._std_vel = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Start a track from an (cx, cy, a, h) measurement."""
        mean = np.zeros(8)
        mean[:4] = measurement
        h = measurement[3]
        std = [
            2 * self._std_pos * h,
            2 * self._std_pos * h,
            1e-2,
            2 * self._std_pos * h,
            10 * self._std_vel * h,
            10 * self._std_vel * h,
            1e-5,
            10 * self._std_vel * h,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [
            self._std_pos * h,
            self._std_pos * h,
            1e-2,
            self._std_pos * h,
            self._std_vel * h,
            self._std_vel * h,
            1e-5,
            self._std_vel * h,
        ]
        Q = np.diag(np.square(std))
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + Q
        return mean, cov

    def update(
        self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h]
        R = np.diag(np.square(std))
        S = self._H @ cov @ self._H.T + R
        K = cov @ self._H.T @ np.linalg.inv(S)
        innov = measurement - self._H @ mean
        mean = mean + K @ innov
        cov = (np.eye(8) - K @ self._H) @ cov
        return mean, cov


def xyxy_to_cxcyah(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float64)
    w = max(x2 - x1, 1e-3)
    h = max(y2 - y1, 1e-3)
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h])


def cxcyah_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, a, h = state[:4]
    h = max(float(h), 1e-3)
    w = max(float(a) * h, 1e-3)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
