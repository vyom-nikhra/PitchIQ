"""Projective-geometry primitives: homography fitting/apply, line math.

A homography here always maps *pixel* coordinates to *pitch* coordinates
(metres) unless stated otherwise. Implemented in pure NumPy so the geometry is
unit-testable without OpenCV; the calibration module layers cv2's RANSAC on top.
"""

from __future__ import annotations

import numpy as np


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N, 2) points. Returns (N, 2).

    Points at infinity (w ~ 0) come back as NaN rather than exploding.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    ones = np.ones((pts.shape[0], 1))
    homog = np.hstack([pts, ones]) @ np.asarray(H, dtype=np.float64).T
    w = homog[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = homog[:, :2] / w
    out[np.abs(w[:, 0]) < 1e-12] = np.nan
    return out


def fit_homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct Linear Transform with Hartley normalisation.

    Exact for 4 correspondences, least-squares for more. Raises ``ValueError``
    on degenerate input (fewer than 4 points or collinear configurations).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] < 4 or src.shape != dst.shape:
        raise ValueError(f"need >=4 matched points, got {src.shape} vs {dst.shape}")

    def normalise(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = pts.mean(axis=0)
        d = np.sqrt(((pts - mean) ** 2).sum(axis=1)).mean()
        if d < 1e-9:
            raise ValueError("degenerate (coincident) points")
        s = np.sqrt(2) / d
        T = np.array([[s, 0, -s * mean[0]], [0, s, -s * mean[1]], [0, 0, 1]])
        p = (np.hstack([pts, np.ones((len(pts), 1))]) @ T.T)[:, :2]
        return p, T

    s_n, Ts = normalise(src)
    d_n, Td = normalise(dst)
    A = []
    for (x, y), (u, v) in zip(s_n, d_n):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, S, Vt = np.linalg.svd(np.asarray(A))
    if S[-2] < 1e-10:  # rank deficiency -> collinear input
        raise ValueError("degenerate point configuration")
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("ill-conditioned homography")
    return H / H[2, 2]


def reprojection_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    """RMS error (in dst units) of ``H @ src`` against ``dst``."""
    proj = apply_homography(H, src)
    err = np.linalg.norm(proj - np.asarray(dst, dtype=np.float64), axis=1)
    return float(np.sqrt(np.nanmean(err**2)))


def line_intersection(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> np.ndarray | None:
    """Intersection of infinite lines (p1,p2) and (p3,p4); None if parallel."""
    p1, p2, p3, p4 = (np.asarray(p, dtype=np.float64) for p in (p1, p2, p3, p4))
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    return p1 + t * d1


def segment_angle_deg(seg: np.ndarray) -> float:
    """Orientation of a segment [x1,y1,x2,y2] in [0, 180) degrees."""
    x1, y1, x2, y2 = np.asarray(seg, dtype=np.float64)
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
    return float(ang)


def point_line_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Perpendicular distance of (N,2) points to the infinite line through a, b."""
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    n = np.hypot(*d)
    if n < 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    return np.abs(np.cross(np.broadcast_to(d, pts.shape), pts - a)) / n


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest difference between two undirected line orientations (deg)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)
