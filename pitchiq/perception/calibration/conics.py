"""Projective conic utilities for circle-based calibration.

The centre circle and penalty arcs image as ellipses. Conic geometry gives
*projectively exact* point correspondences that break the degeneracy of
centre-of-pitch views (where every line×line intersection lies on the halfway
line, so DLT is impossible from lines alone):

* the ellipse ∩ halfway-line points ↔ the circle's top/bottom keypoints,
* the tangency points of the two tangents drawn from the halfway×touchline
  intersection ↔ the same construction on the world circle (tangency is a
  projective invariant),
* the pole of a marking line w.r.t. the conic ↔ the pole of that line w.r.t.
  the world circle (pole/polar duality is projective-invariant).

All conics are 3x3 symmetric matrices Q with x^T Q x = 0 for homogeneous x.
"""

from __future__ import annotations

import numpy as np


def ellipse_to_conic(cx: float, cy: float, major: float, minor: float, angle_deg: float) -> np.ndarray:
    """Conic matrix of an ellipse given in OpenCV ``fitEllipse`` convention
    (centre, FULL axis lengths, rotation of the *first* axis in degrees)."""
    a = major / 2.0
    b = minor / 2.0
    th = np.radians(angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    # x'^T D x' = 1 in ellipse frame; D = diag(1/a², 1/b²)
    D = np.diag([1.0 / a**2, 1.0 / b**2])
    A = R @ D @ R.T
    c = np.array([cx, cy])
    Q = np.zeros((3, 3))
    Q[:2, :2] = A
    Q[:2, 2] = -A @ c
    Q[2, :2] = -A @ c
    Q[2, 2] = c @ A @ c - 1.0
    return Q


def circle_conic(cx: float, cy: float, r: float) -> np.ndarray:
    """Conic matrix of a world circle."""
    return ellipse_to_conic(cx, cy, 2 * r, 2 * r, 0.0)


def polar_line(Q: np.ndarray, point_xy: np.ndarray) -> np.ndarray:
    """Polar line (homogeneous 3-vector) of a point w.r.t. conic Q."""
    p = np.array([point_xy[0], point_xy[1], 1.0])
    return Q @ p


def pole_point(Q: np.ndarray, line: np.ndarray) -> np.ndarray | None:
    """Pole (inhomogeneous 2-vector) of a line w.r.t. conic Q."""
    try:
        p = np.linalg.solve(Q, line)
    except np.linalg.LinAlgError:
        return None
    if abs(p[2]) < 1e-12:
        return None
    return p[:2] / p[2]


def line_through(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Homogeneous line through two inhomogeneous points."""
    a = np.array([p1[0], p1[1], 1.0])
    b = np.array([p2[0], p2[1], 1.0])
    return np.cross(a, b)


def conic_line_intersections(Q: np.ndarray, line: np.ndarray) -> list[np.ndarray]:
    """Intersect a conic with a homogeneous line. Returns 0-2 real points."""
    l = np.asarray(line, dtype=np.float64)
    # parameterise the line by two points on it
    if abs(l[0]) >= abs(l[1]):
        p0 = np.array([-l[2] / l[0], 0.0, 1.0])
        d = np.array([-l[1] / l[0], 1.0, 0.0])
    else:
        p0 = np.array([0.0, -l[2] / l[1], 1.0])
        d = np.array([1.0, -l[0] / l[1], 0.0])
    # (p0 + t d)^T Q (p0 + t d) = 0
    a = d @ Q @ d
    b = 2.0 * (p0 @ Q @ d)
    c = p0 @ Q @ p0
    if abs(a) < 1e-14:
        if abs(b) < 1e-14:
            return []
        ts = [-c / b]
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return []
        s = np.sqrt(disc)
        ts = [(-b - s) / (2 * a), (-b + s) / (2 * a)]
    out = []
    for t in ts:
        p = p0 + t * d
        if abs(p[2]) > 1e-12:
            out.append(p[:2] / p[2])
    return out


def tangent_points_from(Q: np.ndarray, external_pt: np.ndarray) -> list[np.ndarray]:
    """Tangency points of the two tangents from an external point to conic Q
    (= polar(point) ∩ Q). Projectively invariant construction."""
    return conic_line_intersections(Q, polar_line(Q, external_pt))


def sample_ellipse(cx: float, cy: float, major: float, minor: float, angle_deg: float,
                   n: int = 90) -> np.ndarray:
    """(n, 2) points along the ellipse (for mask-coverage validation)."""
    a, b = major / 2.0, minor / 2.0
    th = np.radians(angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([a * np.cos(t), b * np.sin(t)], axis=1) @ R.T
    return pts + np.array([cx, cy])
