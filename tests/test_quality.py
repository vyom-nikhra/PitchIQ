"""Perception-quality assessment tests: level grading, notes, edge cases."""

import numpy as np
import pandas as pd

from pitchiq.core.schema import BALL_ID, MatchMeta
from pitchiq.report.quality import assess_quality


def _cv_meta(n_frames: int = 200, extras: dict | None = None) -> MatchMeta:
    return MatchMeta(fps=25, n_frames=n_frames,
                     source="video:clip.mp4|detector:yolo:weights/x.pt",
                     extras=extras or {})


def _tracking(n_frames: int = 200, n_players: int = 22,
              ball_frames: list[int] | None = None,
              ball_conf: float = 0.9) -> pd.DataFrame:
    rows = []
    for f in range(n_frames):
        for pid in range(1, n_players + 1):
            rows.append(dict(frame=f, timestamp=f / 25.0, entity_id=pid,
                             conf=0.9, x_pitch=50.0, y_pitch=30.0))
        if ball_frames is None or f in ball_frames:
            rows.append(dict(frame=f, timestamp=f / 25.0, entity_id=BALL_ID,
                             conf=ball_conf, x_pitch=50.0, y_pitch=30.0))
    return pd.DataFrame(rows)


def _homography(n_frames: int = 200, method: str = "keypoints",
                reproj: float = 5.0) -> pd.DataFrame:
    h = {f"h{i}{j}": 1.0 for i in range(3) for j in range(3)}
    return pd.DataFrame([dict(frame=f, timestamp=f / 25.0, **h,
                              reproj_error_px=reproj, method=method,
                              is_scene_cut=False) for f in range(n_frames)])


def test_ground_truth_source_is_high_with_note():
    meta = MatchMeta(fps=25, n_frames=100, source="synthetic-simulator")
    q = assess_quality(pd.DataFrame(), None, meta)
    assert q["overall"] == "high"
    assert q["is_cv"] is False
    assert "ground truth" in q["notes"][0]


def test_good_cv_job_grades_high():
    meta = _cv_meta(extras={"team_separability": 4.0,
                            "ball_observed_frames": 200})
    q = assess_quality(_tracking(), _homography(), meta)
    assert q["levels"] == {"calibration": "high", "ball": "high",
                          "tracking": "high", "teams": "high"}
    assert q["overall"] == "high"
    assert q["components"]["ball_observed_is_estimated"] is False


def test_sparse_ball_drags_overall_low():
    meta = _cv_meta(extras={"team_separability": 4.0,
                            "ball_observed_frames": 20})
    q = assess_quality(_tracking(), _homography(), meta)
    assert q["levels"]["ball"] == "low"
    assert q["overall"] == "low"
    assert any("observed in only" in n for n in q["notes"])


def test_missing_homography_grades_calibration_low():
    meta = _cv_meta(extras={"team_separability": 4.0,
                            "ball_observed_frames": 200})
    q = assess_quality(_tracking(), None, meta)
    assert q["levels"]["calibration"] == "low"
    assert q["overall"] == "low"
    assert any("calibration" in n for n in q["notes"])


def test_estimated_observed_share_excludes_interpolated_rows():
    # 150 observed frames at conf 0.8, 50 interpolation-bridged at conf 0.15
    # (interpolate_ball decays conf to <=0.5x the neighbours)
    df = _tracking(ball_frames=list(range(150)), ball_conf=0.8)
    interp = pd.DataFrame([dict(frame=f, timestamp=f / 25.0, entity_id=BALL_ID,
                                conf=0.15, x_pitch=50.0, y_pitch=30.0)
                           for f in range(150, 200)])
    df = pd.concat([df, interp], ignore_index=True)
    meta = _cv_meta(extras={"team_separability": 4.0})  # no exact count
    q = assess_quality(df, _homography(), meta)
    assert q["components"]["ball_observed_is_estimated"] is True
    assert abs(q["components"]["ball_observed_share"] - 0.75) < 0.02
    assert q["levels"]["ball"] == "high"


def test_flow_only_calibration_error_ignores_flow_frames():
    # flow rows carry a propagated (near-zero) residual that must not be
    # mistaken for measured accuracy: error comes from direct solves only
    h = pd.concat([_homography(20, method="lines", reproj=12.0),
                   _homography(180, method="flow", reproj=0.0)],
                  ignore_index=True)
    meta = _cv_meta(extras={"team_separability": 4.0,
                            "ball_observed_frames": 200})
    q = assess_quality(_tracking(), h, meta)
    assert q["components"]["calibration_reproj_px_median"] == 12.0
    assert any("optical-flow" in n for n in q["notes"])


def test_short_tracks_and_low_separability_noted():
    df = _tracking(n_players=8)
    # fragment identities: give every 20-frame window fresh track ids
    df.loc[df.entity_id != BALL_ID, "entity_id"] = (
        df.loc[df.entity_id != BALL_ID, "entity_id"]
        + 100 * (df.loc[df.entity_id != BALL_ID, "frame"] // 20))
    meta = _cv_meta(extras={"team_separability": 1.0,
                            "ball_observed_frames": 200})
    q = assess_quality(df, _homography(), meta)
    assert q["levels"]["tracking"] == "low"
    assert q["levels"]["teams"] == "low"
    assert any("short-lived" in n for n in q["notes"])
    assert any("Kit colours" in n for n in q["notes"])


def test_empty_ball_is_low_not_crash():
    meta = _cv_meta(extras={"team_separability": 4.0})
    q = assess_quality(_tracking(ball_frames=[]), _homography(), meta)
    assert q["levels"]["ball"] == "low"
    assert np.isfinite(q["components"]["ball_frame_coverage"])
