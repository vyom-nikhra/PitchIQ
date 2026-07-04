"""Frame-to-frame camera-motion estimation (BoT-SORT's GMC, simplified).

Broadcast cameras pan and zoom constantly; without compensation the Kalman
prediction lags every pan and IoU association degrades. We estimate a partial
affine (rotation+scale+translation) between consecutive frames from sparse
optical flow on *background* features (detected boxes are masked out so
players don't bias the estimate). The same affine also propagates the pitch
homography between full re-estimations (see calibration.temporal).
"""

from __future__ import annotations

import cv2
import numpy as np


class CameraMotionEstimator:
    def __init__(self, max_corners: int = 400, quality: float = 0.01, min_distance: int = 12) -> None:
        self.prev_gray: np.ndarray | None = None
        self.feature_params = dict(
            maxCorners=max_corners, qualityLevel=quality, minDistance=min_distance, blockSize=7
        )
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def reset(self) -> None:
        self.prev_gray = None

    def estimate(
        self, frame_bgr: np.ndarray, exclude_boxes: list[np.ndarray] | None = None
    ) -> np.ndarray | None:
        """Affine (2x3) mapping previous-frame pixels to current-frame pixels,
        or None on the first frame / estimation failure."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=0.5, fy=0.5)  # speed; affine rescaled below
        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        mask = np.full(self.prev_gray.shape, 255, dtype=np.uint8)
        for bb in exclude_boxes or []:
            x1, y1, x2, y2 = (np.asarray(bb) * 0.5).astype(int)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

        pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)
        A = None
        if pts is not None and len(pts) >= 12:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None, **self.lk_params)
            good = status.ravel() == 1
            if good.sum() >= 12:
                A_half, inliers = cv2.estimateAffinePartial2D(
                    pts[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=3.0
                )
                if A_half is not None and inliers is not None and inliers.sum() >= 10:
                    # rescale the affine from half-res to full-res coordinates
                    S = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]])
                    Sinv = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1.0]])
                    A3 = np.vstack([A_half, [0, 0, 1]])
                    A = (S @ A3 @ Sinv)[:2]
        self.prev_gray = gray
        return A
