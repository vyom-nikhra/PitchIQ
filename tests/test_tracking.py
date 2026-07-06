import numpy as np
import pandas as pd

from pitchiq.config import TrackingConfig
from pitchiq.core.types import Detection, EntityClass, iou_matrix
from pitchiq.perception.tracking import ByteTracker, STrack
from pitchiq.perception.tracking.metrics import evaluate_tracking


def _det(x, y, w=20, h=40, conf=0.9, cls=EntityClass.PLAYER):
    return Detection(bbox=np.array([x, y, x + w, y + h], dtype=np.float32), conf=conf, cls=cls)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
    iou = iou_matrix(a, b)
    assert iou.shape == (1, 3)
    assert iou[0, 0] == 1.0
    assert 0.1 < iou[0, 1] < 0.2
    assert iou[0, 2] == 0.0


def test_ids_persist_through_linear_motion():
    tracker = ByteTracker(TrackingConfig())
    ids_per_frame = []
    for f in range(30):
        dets = [_det(100 + 3 * f, 100), _det(400 - 3 * f, 200)]
        tracks = tracker.update(dets)
        ids_per_frame.append(sorted(t.track_id for t in tracks))
    assert ids_per_frame[-1] == ids_per_frame[1]  # same two ids throughout
    assert len(set(ids_per_frame[-1])) == 2


def test_id_survives_short_occlusion():
    tracker = ByteTracker(TrackingConfig())
    id_before = None
    for f in range(40):
        if 15 <= f < 25:
            dets = []  # occluded
        else:
            dets = [_det(100 + 4 * f, 150)]
        tracks = tracker.update(dets)
        if f == 14:
            id_before = tracks[0].track_id
        if f >= 26 and tracks:
            assert tracks[0].track_id == id_before
    assert id_before is not None


def test_low_conf_detections_keep_track_alive():
    """The ByteTrack signature move: second-stage association on low scores."""
    tracker = ByteTracker(TrackingConfig())
    for f in range(30):
        conf = 0.9 if f < 10 or f > 20 else 0.25  # dips below high_thresh
        tracks = tracker.update([_det(100 + 3 * f, 150, conf=conf)])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_velocity_gate_rejects_impossible_association():
    """The pitch-space velocity gate must reject a (track, det) pair whose
    implied real-pitch speed exceeds the cap, while keeping a plausible one."""
    import numpy as np

    from pitchiq.core.geometry import fit_homography_dlt

    # pixel->pitch: 10 px == 1 m in the sampled region
    src = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], float)
    dst = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], float)
    H = fit_homography_dlt(src, dst)
    cfg = TrackingConfig(max_assoc_speed_mps=11.0)
    tracker = ByteTracker(cfg)
    tracker._H = H
    tracker._dt = 0.04  # 25 fps

    track = STrack(_det(500, 250))
    track._gate_foot = (510.0, 290.0)   # last observed foot (box bottom-centre)
    track._gate_dt = 0.04
    # _det(x, y) box is [x, y, x+20, y+40] so its foot is (x+10, y+40).
    near = _det(500, 250)   # foot (510, 290): 0 m displacement — allowed
    far = _det(800, 250)    # foot (810, 290): 300 px ≈ 30 m in 0.04 s — impossible

    cost = tracker._cost([track], [near, far], use_appearance=False)
    assert cost[0, 0] < 1e4          # plausible pair kept
    assert cost[0, 1] >= 1e4         # impossible pair rejected

    # without a homography the gate is a no-op (pixel IoU only)
    tracker._H = None
    cost2 = tracker._cost([track], [near, far], use_appearance=False)
    assert cost2[0, 1] < 1e4


def test_ball_excluded():
    tracker = ByteTracker(TrackingConfig())
    tracks = tracker.update([_det(10, 10, cls=EntityClass.BALL)])
    assert tracks == []


def test_metrics_perfect_and_switch():
    rows = []
    for f in range(20):
        rows.append(dict(frame=f, track_id=1, x1=0 + f, y1=0, x2=20 + f, y2=40))
    gt = pd.DataFrame(rows)
    perfect = evaluate_tracking(gt, gt.copy())
    assert perfect["mota"] == 1.0
    assert perfect["idf1"] == 1.0
    assert perfect["id_switches"] == 0

    # identity swap halfway through -> 1 switch, IDF1 well below 1
    pred = gt.copy()
    pred.loc[pred.frame >= 10, "track_id"] = 2
    res = evaluate_tracking(gt, pred)
    assert res["id_switches"] == 1
    assert res["mota"] < 1.0
    assert 0.4 < res["idf1"] <= 0.5 + 1e-9
