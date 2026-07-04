"""Chamfer refinement of a homography against the white-line mask.

Constructed-point solves (line intersections, conic poles) use a handful of
correspondences, so small pixel noise in any of them shears the solution.
This module polishes an already-plausible homography with *all* the evidence:
project the full pitch template into the image and minimise the distance
transform of the marking mask at those points (classic chamfer alignment)
over the 8 homography parameters, with a soft-L1 loss so occluded/missing
template parts don't drag the fit.

The optimisation is local — the seed must already be roughly right (it is:
hypothesis scoring only passes solutions whose projected template overlaps
the mask). A trust guard rejects refinements that wander.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import optimize

from pitchiq.core.geometry import apply_homography


def _bilinear(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample of a float image at (N,2) positions (x, y)."""
    h, w = img.shape[:2]
    x = np.clip(xy[:, 0], 0, w - 1.001)
    y = np.clip(xy[:, 1], 0, h - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = x - x0
    fy = y - y0
    v00 = img[y0, x0]
    v01 = img[y0, x0 + 1]
    v10 = img[y0 + 1, x0]
    v11 = img[y0 + 1, x0 + 1]
    return (v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy)
            + v10 * (1 - fx) * fy + v11 * fx * fy)


def refine_homography(
    H: np.ndarray,
    mask: np.ndarray,
    template_pts: np.ndarray,
    max_nfev: int = 60,
    trust_frac: float = 0.08,
) -> tuple[np.ndarray, float, float] | None:
    """Refine pixel→pitch ``H`` against the marking ``mask``.

    Returns ``(H_refined, cost_before, cost_after)`` or None if the seed is
    unusable. ``template_pts``: (N,2) world-metre samples of all markings.
    ``trust_frac``: max mean control-point drift as a fraction of the frame
    diagonal before the refinement is rejected as having wandered.
    """
    h_img, w_img = mask.shape[:2]
    try:
        G0 = np.linalg.inv(np.asarray(H, dtype=np.float64))  # world -> image
    except np.linalg.LinAlgError:
        return None

    dt = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3).astype(np.float32)
    dt = np.minimum(dt, 25.0)  # cap: far-away template parts saturate, not explode

    # world normalisation so the 8 update params are comparably scaled
    Tw = np.array([[1 / 50.0, 0, -52.5 / 50.0], [0, 1 / 34.0, -1.0], [0, 0, 1]])
    Tw_inv = np.linalg.inv(Tw)

    # subsample template for speed
    pts = template_pts
    if len(pts) > 420:
        pts = pts[:: len(pts) // 420]

    def G_of(p: np.ndarray) -> np.ndarray:
        dP = np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 0.0]])
        return G0 @ Tw_inv @ (np.eye(3) + dP) @ Tw

    def residuals(p: np.ndarray) -> np.ndarray:
        proj = apply_homography(G_of(p), pts)
        ok = np.isfinite(proj).all(axis=1)
        inside = ok & (proj[:, 0] >= 0) & (proj[:, 0] < w_img - 1) \
                    & (proj[:, 1] >= 0) & (proj[:, 1] < h_img - 1)
        r = np.full(len(pts), 25.0, dtype=np.float64)
        if inside.any():
            r[inside] = _bilinear(dt, proj[inside])
        # points legitimately outside the frame shouldn't be punished
        r[~ok] = 25.0
        outside_frame = ok & ~inside
        r[outside_frame] = 0.0
        return r

    r0 = residuals(np.zeros(8))
    visible0 = r0 < 25.0
    if visible0.sum() < 40:
        return None
    cost_before = float(np.mean(r0[visible0]))

    try:
        sol = optimize.least_squares(
            residuals, np.zeros(8), method="trf", loss="soft_l1", f_scale=3.0,
            max_nfev=max_nfev, xtol=1e-10,
        )
    except Exception:
        return None

    G1 = G_of(sol.x)
    try:
        H1 = np.linalg.inv(G1)
        H1 /= H1[2, 2]
    except np.linalg.LinAlgError:
        return None

    # trust guard: control points must not wander
    ctrl_w = np.array([[21.0, 13.6], [84.0, 13.6], [84.0, 54.4], [21.0, 54.4]])
    before = apply_homography(G0, ctrl_w)
    after = apply_homography(G1, ctrl_w)
    if not (np.all(np.isfinite(before)) and np.all(np.isfinite(after))):
        return None
    drift = float(np.linalg.norm(after - before, axis=1).mean())
    if drift > trust_frac * np.hypot(h_img, w_img):
        return None

    r1 = residuals(sol.x)
    visible1 = r1 < 25.0
    cost_after = float(np.mean(r1[visible1])) if visible1.any() else cost_before
    if cost_after >= cost_before:
        return None
    return H1, cost_before, cost_after
