import numpy as np

from pitchiq.core.schema import BALL_ID, validate_tracking_table


def test_tracking_table_valid(sim_result):
    df = validate_tracking_table(sim_result.tracking)
    # 23 entities per frame: 2x11 + referee + ball = 24 rows
    per_frame = df.groupby("frame").size()
    assert (per_frame == 24).all()
    assert set(df["class"].unique()) == {"player", "goalkeeper", "referee", "ball"}


def test_positions_on_pitch(sim_result):
    df = sim_result.tracking
    players = df[df["class"] != "ball"]
    assert players["x_pitch"].between(-2, 107).all()
    assert players["y_pitch"].between(-2, 70).all()


def test_realistic_speeds(sim_result):
    df = sim_result.tracking
    fps = sim_result.meta.fps
    halftime = sim_result.meta.extras["halftime_frame"]
    speeds = []
    for eid, g in df[df["class"] == "player"].groupby("entity_id"):
        g = g.sort_values("frame")
        v = np.hypot(np.diff(g.x_pitch), np.diff(g.y_pitch)) * fps
        frames = g.frame.to_numpy()[1:]
        # exclude kickoff/halftime teleports
        v = v[(np.abs(frames - halftime) > 2) & (frames > 2)]
        speeds.append(v)
    v = np.concatenate(speeds)
    assert np.percentile(v, 99) < 12.0
    assert 0.5 < np.median(v) < 5.0


def test_pass_events_sane(sim_result):
    ev = sim_result.events
    passes = ev[ev.type == "pass"]
    minutes = sim_result.meta.n_frames / sim_result.meta.fps / 60
    assert len(passes) / minutes > 5, "too few passes simulated"
    completion = (passes.outcome == "complete").mean()
    assert 0.4 < completion < 0.98
    assert passes[["x", "y", "end_x", "end_y"]].notna().all().all()


def test_possession_balanced(sim_result):
    poss = sim_result.possession_gt
    home_share = float((poss == "home").mean())
    assert 0.25 < home_share < 0.9


def test_man_marking_ground_truth(sim_result):
    # the demo fixture's away team man-marks
    assert "away" in sim_result.marking_gt
    pairs = sim_result.marking_gt["away"]
    assert len(pairs) == 10
    # marked targets are distinct home outfielders
    assert len(set(pairs.values())) == 10


def test_ball_always_present(sim_result):
    ball = sim_result.tracking[sim_result.tracking.entity_id == BALL_ID]
    assert ball.frame.nunique() == sim_result.meta.n_frames
