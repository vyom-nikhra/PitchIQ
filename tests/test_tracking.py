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


def _det_feat(x, y, feat, conf=0.9):
    d = _det(x, y, conf=conf)
    d.feature = np.asarray(feat, dtype=np.float32)
    return d


def _reid_setup():
    """Tracker + homography (10 px = 1 m) with two players tracked pre-cut."""
    from pitchiq.core.geometry import fit_homography_dlt

    src = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], float)
    dst = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], float)
    H = fit_homography_dlt(src, dst)
    cfg = TrackingConfig()
    cfg.appearance.enabled = True
    tracker = ByteTracker(cfg)
    fa = np.array([1.0] + [0.0] * 7, np.float32)   # player A's appearance
    fb = np.array([0.0, 1.0] + [0.0] * 6, np.float32)
    for f in range(8):
        tracker.update([_det_feat(100 + 2 * f, 100, fa),
                        _det_feat(600, 300, fb)], homography=H, dt=0.04)
    ids = {t.track_id for t in tracker.tracked}
    return tracker, H, fa, fb, ids


def test_cross_cut_reid_restores_identity():
    """After a scene cut, a track with matching appearance near the last
    pitch position must inherit the pre-cut ID."""
    tracker, H, fa, fb, ids_before = _reid_setup()
    id_a = next(t.track_id for t in tracker.tracked
                if t.feature is not None and t.feature[0] > 0.9)

    # cut: player A reappears at a different PIXEL location that projects to
    # a nearby pitch point (camera angle changed, player barely moved)
    out = []
    for f in range(3):
        out = tracker.update([_det_feat(150 + f, 110, fa)],
                             homography=H, dt=0.04, scene_cut=(f == 0))
    assert len(out) == 1
    assert out[0].track_id == id_a  # identity restored across the cut


def test_cross_cut_reid_rejects_stranger_and_far_player():
    """A post-cut track with an unfamiliar appearance gets a FRESH id, and a
    familiar appearance too far away on the pitch is also refused."""
    tracker, H, fa, fb, ids_before = _reid_setup()

    fz = np.array([0.0, 0.0, 1.0] + [0.0] * 5, np.float32)  # stranger
    out = []
    for f in range(3):
        out = tracker.update([_det_feat(100, 100, fz)],
                             homography=H, dt=0.04, scene_cut=(f == 0))
    assert out and out[0].track_id not in ids_before

    tracker2, H2, fa2, fb2, ids2 = _reid_setup()
    out2 = []
    for f in range(3):
        out2 = tracker2.update([_det_feat(0, 30, fb2)],  # ~(1m, 7m): B was at (61m, 34m)
                               homography=H2, dt=0.04, scene_cut=(f == 0))
    assert out2 and out2[0].track_id not in ids2  # right shirt, wrong place


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
