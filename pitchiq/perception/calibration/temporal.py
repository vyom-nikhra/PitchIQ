"""Temporal calibration machinery: scene cuts, smoothing, flow propagation.

Broadcast footage pans/zooms continuously and hard-cuts between cameras:

* :class:`SceneCutDetector` — HSV-histogram Bhattacharyya distance between
  consecutive frames; a cut resets all temporal state (fresh full estimation).
* :class:`HomographySmoother` — homographies are smoothed *in point space*:
  we EMA the image projections of four fixed pitch control points and refit an
  exact 4-point DLT. Smoothing raw matrix entries is numerically meaningless
  (they live on a projective manifold); control points are stable and
  interpretable. Also resolves the pitch's mirror symmetry: line-based
  calibration cannot distinguish a view from its 180° twin, so each new
  estimate is canonicalised against the running solution by testing the four
  symmetry composes and keeping the closest — orientation stays consistent
  within a scene.
* flow propagation — between full estimations (or when estimation fails) the
  previous homography is composed with the frame-to-frame camera affine
  estimated by :class:`~pitchiq.perception.tracking.camera_motion.CameraMotionEstimator`:
  ``H_t = H_{t-1} ∘ A_t^{-1}``.
"""

from __future__ import annotations

import cv2
import numpy as np

from pitchiq.core.geometry import apply_homography, fit_homography_dlt
from pitchiq.core.pitch import Pitch


class SceneCutDetector:
    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold
        self._prev_hist: np.ndarray | None = None

    def __call__(self, frame_bgr: np.ndarray) -> bool:
        hsv = cv2.cvtColor(cv2.resize(frame_bgr, (160, 90)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        is_cut = False
        if self._prev_hist is not None:
            d = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            is_cut = bool(d > self.threshold)
        self._prev_hist = hist
        return is_cut


def _symmetry_transforms(pitch: Pitch) -> list[np.ndarray]:
    """Pitch-coordinate maps under which the marking template is invariant."""
    L, W = pitch.length, pitch.width
    I3 = np.eye(3)
    mx = np.array([[-1, 0, L], [0, 1, 0], [0, 0, 1]], dtype=float)
    my = np.array([[1, 0, 0], [0, -1, W], [0, 0, 1]], dtype=float)
    return [I3, mx, my, mx @ my]


class HomographySmoother:
    def __init__(self, pitch: Pitch, alpha: float = 0.5) -> None:
        self.pitch = pitch
        self.alpha = alpha
        # control points well inside the pitch, non-collinear
        L, W = pitch.length, pitch.width
        self.ctrl_pitch = np.array(
            [[0.2 * L, 0.2 * W], [0.8 * L, 0.2 * W], [0.8 * L, 0.8 * W], [0.2 * L, 0.8 * W]]
        )
        self._ctrl_img: np.ndarray | None = None
        self._sym = _symmetry_transforms(pitch)

    def reset(self) -> None:
        self._ctrl_img = None

    @property
    def current(self) -> np.ndarray | None:
        if self._ctrl_img is None:
            return None
        return fit_homography_dlt(self._ctrl_img, self.ctrl_pitch)

    def _img_points(self, H: np.ndarray) -> np.ndarray | None:
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        pts = apply_homography(Hinv, self.ctrl_pitch)
        return pts if np.all(np.isfinite(pts)) else None

    def canonicalize(self, H: np.ndarray) -> np.ndarray:
        """Resolve mirror ambiguity: pick the symmetry compose of ``H`` whose
        control points land closest to the running smoothed solution."""
        if self._ctrl_img is None:
            return H
        best_H, best_d = H, np.inf
        for S in self._sym:
            Hs = S @ H
            pts = self._img_points(Hs)
            if pts is None:
                continue
            d = float(np.linalg.norm(pts - self._ctrl_img, axis=1).mean())
            if d < best_d:
                best_d, best_H = d, Hs
        return best_H

    def update(self, H_new: np.ndarray, hard: bool = False) -> np.ndarray:
        """Blend a fresh estimate into the smoothed solution and return it.

        ``hard=True`` (first frame after a cut) snaps instead of blending.
        """
        H_new = self.canonicalize(H_new)
        pts = self._img_points(H_new)
        if pts is None:
            return self.current if self._ctrl_img is not None else H_new
        if self._ctrl_img is None or hard:
            self._ctrl_img = pts
        else:
            # reject wild jumps (mis-calibration): >25% frame-diagonal shift
            jump = float(np.linalg.norm(pts - self._ctrl_img, axis=1).mean())
            scale = float(np.linalg.norm(self._ctrl_img[0] - self._ctrl_img[2]) + 1e-6)
            if jump > 0.6 * scale:
                return self.current
            self._ctrl_img = self.alpha * pts + (1 - self.alpha) * self._ctrl_img
        return self.current

    def propagate(self, affine_2x3: np.ndarray) -> np.ndarray | None:
        """Advance the smoothed solution by a camera affine (prev px → cur px)."""
        if self._ctrl_img is None:
            return None
        A = np.vstack([affine_2x3, [0, 0, 1]])
        pts = (np.hstack([self._ctrl_img, np.ones((4, 1))]) @ A.T)[:, :2]
        if not np.all(np.isfinite(pts)):
            return self.current
        self._ctrl_img = pts
        return self.current
