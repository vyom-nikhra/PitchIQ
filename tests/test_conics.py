import numpy as np

from pitchiq.perception.calibration.conics import (
    circle_conic,
    conic_line_intersections,
    ellipse_to_conic,
    line_through,
    pole_point,
    polar_line,
    sample_ellipse,
    tangent_points_from,
)


def _on_conic(Q, p, tol=1e-6):
    v = np.array([p[0], p[1], 1.0])
    return abs(v @ Q @ v) < tol


def test_ellipse_conic_contains_its_points():
    Q = ellipse_to_conic(10, 5, 8, 4, 30)
    for p in sample_ellipse(10, 5, 8, 4, 30, 24):
        assert _on_conic(Q, p, tol=1e-9)


def test_conic_line_intersections_circle():
    Q = circle_conic(0, 0, 5)
    pts = conic_line_intersections(Q, np.array([0.0, 1.0, 0.0]))  # y = 0
    xs = sorted(p[0] for p in pts)
    assert np.allclose(xs, [-5, 5])


def test_pole_polar_duality():
    Q = circle_conic(2, 3, 4)
    line = np.array([1.0, 0.5, -7.0])
    pole = pole_point(Q, line)
    back = polar_line(Q, pole)
    assert np.allclose(back / np.linalg.norm(back), line / np.linalg.norm(line)) or np.allclose(
        back / np.linalg.norm(back), -line / np.linalg.norm(line)
    )


def test_tangent_points_from_external():
    Q = circle_conic(0, 0, 1)
    ts = tangent_points_from(Q, np.array([2.0, 0.0]))
    assert len(ts) == 2
    for t in ts:
        assert _on_conic(Q, t)
        # tangency: the tangent from (2,0) touches at x = 1/2
        assert abs(t[0] - 0.5) < 1e-9


def test_pole_of_box_front_wrt_penalty_arc():
    """The construction used by arc calibration: known closed form."""
    Q = circle_conic(94.0, 34.0, 9.15)
    pole = pole_point(Q, np.array([1.0, 0.0, -88.5]))  # line x = 88.5
    # pole = (cx - r^2/(cx - 88.5), cy)
    assert np.allclose(pole, [94.0 - 9.15**2 / 5.5, 34.0])


def test_line_through():
    l = line_through(np.array([0, 0]), np.array([1, 1]))
    for p in ([0, 0, 1], [1, 1, 1], [5, 5, 1]):
        assert abs(np.dot(l, p)) < 1e-9
