import numpy as np

from pitchiq.core.pitch import Pitch, to_attacking_coords


def test_keypoints_complete_and_on_pitch(pitch):
    names, kps = pitch.keypoint_array()
    assert len(names) == 33
    assert (kps[:, 0] >= 0).all() and (kps[:, 0] <= pitch.length).all()
    assert (kps[:, 1] >= 0).all() and (kps[:, 1] <= pitch.width).all()


def test_penalty_arc_meets_box_front(pitch):
    """The arc endpoints must lie exactly on the penalty-area front line."""
    for side in ("left", "right"):
        for pos in ("top", "bottom"):
            x, y = pitch.keypoints[f"arc_{side}_{pos}"]
            expected_x = 16.5 if side == "left" else pitch.length - 16.5
            assert abs(x - expected_x) < 1e-9
    # arc endpoints are on the circle around the penalty spot
    sx, sy = pitch.keypoints["penalty_spot_left"]
    ax, ay = pitch.keypoints["arc_left_top"]
    assert abs(np.hypot(ax - sx, ay - sy) - 9.15) < 1e-9


def test_zones(pitch):
    assert pitch.third_of(np.array([10.0, 50.0, 100.0])).tolist() == [0, 1, 2]
    assert pitch.in_penalty_area(np.array([5.0]), np.array([34.0]), "left").all()
    assert not pitch.in_penalty_area(np.array([30.0]), np.array([34.0]), "left").any()
    assert pitch.contains(np.array([52.0]), np.array([34.0])).all()
    assert not pitch.contains(np.array([-10.0]), np.array([34.0])).any()


def test_attacking_coords_flip():
    xy = np.array([[10.0, 10.0]])
    flipped = to_attacking_coords(xy, -1)
    assert np.allclose(flipped, [[95.0, 58.0]])
    assert np.allclose(to_attacking_coords(xy, 1), xy)


def test_sample_line_points_dense(pitch):
    pts = pitch.sample_line_points(step=1.0)
    assert len(pts) > 500
    assert pitch.contains(pts[:, 0], pts[:, 1], margin=0.1).all()
