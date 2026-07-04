import numpy as np
import pytest

from pitchiq.core.geometry import (
    angle_diff_deg,
    apply_homography,
    fit_homography_dlt,
    line_intersection,
    point_line_distance,
    reprojection_error,
)


def test_dlt_exact_recovery():
    """DLT must exactly recover a known homography from 4+ points."""
    rng = np.random.default_rng(0)
    H_true = np.array([[1.2, 0.1, 5.0], [-0.05, 0.9, -3.0], [0.001, 0.0004, 1.0]])
    src = rng.uniform(0, 100, (8, 2))
    dst = apply_homography(H_true, src)
    H = fit_homography_dlt(src, dst)
    assert np.allclose(H / H[2, 2], H_true / H_true[2, 2], atol=1e-6)
    assert reprojection_error(H, src, dst) < 1e-8


def test_dlt_rejects_collinear():
    src = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
    dst = src * 2
    with pytest.raises(ValueError):
        fit_homography_dlt(src, dst)


def test_apply_homography_point_at_infinity():
    H = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float)  # w = x
    out = apply_homography(H, [[0.0, 5.0]])
    assert np.isnan(out).all()


def test_line_intersection():
    p = line_intersection([0, 0], [2, 2], [0, 2], [2, 0])
    assert np.allclose(p, [1, 1])
    assert line_intersection([0, 0], [1, 0], [0, 1], [1, 1]) is None  # parallel


def test_point_line_distance():
    d = point_line_distance(np.array([[0, 1], [0, -2]]), np.array([-5, 0]), np.array([5, 0]))
    assert np.allclose(d, [1, 2])


def test_angle_diff_wraps():
    assert angle_diff_deg(179, 1) == pytest.approx(2.0)
    assert angle_diff_deg(90, 90) == 0
