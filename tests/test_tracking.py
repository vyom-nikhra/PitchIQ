import numpy as np
import pandas as pd

from pitchiq.config import TrackingConfig
from pitchiq.core.types import Detection, EntityClass, iou_matrix
from pitchiq.perception.tracking import ByteTracker
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
