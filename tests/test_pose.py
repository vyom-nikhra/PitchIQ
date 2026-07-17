"""Pose descriptor + sampler tests (pure maths — no model download)."""

import numpy as np

from pitchiq.perception.pose import PoseSampler, pose_descriptor


def _skeleton(lean_dx: float = 0.0, ankle_gap: float = 10.0,
              conf: float = 0.9) -> np.ndarray:
    """Synthetic upright COCO-17 skeleton inside a 40x100 box at origin."""
    k = np.zeros((17, 3))
    k[:, 2] = conf
    k[5] = [12, 20, conf]    # L shoulder
    k[6] = [28, 20, conf]    # R shoulder
    k[9] = [8, 50, conf]     # L wrist
    k[10] = [32, 50, conf]   # R wrist
    k[11] = [14 + lean_dx, 55, conf]   # L hip
    k[12] = [26 + lean_dx, 55, conf]   # R hip
    k[13] = [15, 75, conf]   # knees
    k[14] = [25, 75, conf]
    k[15] = [20 - ankle_gap / 2, 95, conf]  # ankles
    k[16] = [20 + ankle_gap / 2, 95, conf]
    k[0] = [20, 8, conf]     # nose
    return k


BOX = np.array([0.0, 0.0, 40.0, 100.0])


def test_pose_descriptor_geometry():
    upright = pose_descriptor(_skeleton(lean_dx=0.0), BOX)
    leaning = pose_descriptor(_skeleton(lean_dx=18.0), BOX)
    assert upright is not None and leaning is not None
    lean_u, stride_u, arms_u, crouch_u, compact_u = upright
    assert lean_u < 0.05                      # vertical torso
    assert leaning[0] > lean_u + 0.2          # lean detected
    wide = pose_descriptor(_skeleton(ankle_gap=30.0), BOX)
    assert wide[1] > stride_u                 # bigger stride ratio
    assert 0 < crouch_u <= 1.0 and compact_u > 0


def test_pose_descriptor_rejects_occluded_core():
    k = _skeleton()
    k[11, 2] = 0.05  # left hip invisible
    assert pose_descriptor(k, BOX) is None


def test_pose_sampler_finalize_aggregates():
    s = PoseSampler.__new__(PoseSampler)  # skip model load
    s.acc = {}
    for i in range(5):
        d = pose_descriptor(_skeleton(ankle_gap=8 + i), BOX)
        s.acc.setdefault(3, []).append(d)
    s.acc[9] = s.acc[3][:2]  # too few samples -> omitted
    out = s.finalize(min_samples=3)
    assert list(out.entity_id) == [3]
    assert out.iloc[0].n_samples == 5
    assert out.iloc[0]["stride_std"] > 0
    assert {"lean_mean", "crouch_mean", "compact_std"} <= set(out.columns)
