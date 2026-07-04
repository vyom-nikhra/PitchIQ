"""Tests for phases, pressing, shape, field tilt on constructed fixtures."""

import numpy as np
import pandas as pd

from pitchiq.config import PhasesConfig, PressingConfig, PossessionConfig
from pitchiq.core.schema import MatchMeta
from pitchiq.analytics.phases import phase_summary, segment_phases
from pitchiq.analytics.possession import compute_possession
from pitchiq.analytics.pressing import ppda, pressure_events
from pitchiq.analytics.shape import field_tilt, shape_summary, shape_timeseries

FPS = 25.0


def _row(frame, eid, cls, team, x, y):
    return dict(frame=frame, timestamp=frame / FPS, entity_id=eid, **{"class": cls},
                team=team, jersey_no=None, x_pixel=np.nan, y_pixel=np.nan,
                x_pitch=x, y_pitch=y, conf=1.0)


def _meta(n_frames):
    return MatchMeta(fps=FPS, n_frames=n_frames)


def test_phases_buildup_vs_final_third_vs_setpiece():
    rows = []
    for f in range(400):
        wiggle = 0.6 * np.sin(f)  # live ball must move, else it's a dead ball
        if f < 150:
            bx = 15.0 + wiggle  # home build-up (home attacks +x)
        elif f < 300:
            bx = 90.0 + wiggle  # final third
        else:
            bx = 94.0           # static -> set piece
        rows.append(_row(f, 1, "player", "home", bx, 34.0))
        rows.append(_row(f, 2, "player", "away", min(bx + 8, 103), 30.0))
        rows.append(_row(f, -1, "ball", "none", bx, 34.0))
    df = pd.DataFrame(rows)
    poss = compute_possession(df, FPS, PossessionConfig())
    phases = segment_phases(df, poss, _meta(400), PhasesConfig())
    assert (phases[(phases.frame > 20) & (phases.frame < 140)].phase == "build_up").mean() > 0.9
    mid = phases[(phases.frame > 200) & (phases.frame < 290)]
    assert (mid.phase == "final_third_attack").mean() > 0.9
    late = phases[phases.frame > 360]
    assert (late.phase == "set_piece").mean() > 0.8
    summary = phase_summary(phases, FPS)
    assert "share_overall" in summary and summary["share_overall"]


def test_pressure_event_detected():
    rows = []
    kin_rows = []
    for f in range(100):
        hx = 50.0
        px = 58.0 - 6.0 * f / FPS  # presser closing at 6 m/s
        rows.append(_row(f, 1, "player", "home", hx, 34.0))
        rows.append(_row(f, 2, "player", "away", px, 34.0))
        rows.append(_row(f, -1, "ball", "none", hx + 0.3, 34.0))
        kin_rows.append(dict(frame=f, entity_id=1, x=hx, y=34.0, vx=0.0, vy=0.0,
                             speed=0.0, accel=0.0))
        kin_rows.append(dict(frame=f, entity_id=2, x=px, y=34.0, vx=-6.0, vy=0.0,
                             speed=6.0, accel=0.0))
    df = pd.DataFrame(rows)
    kin = pd.DataFrame(kin_rows)
    poss = compute_possession(df, FPS, PossessionConfig())
    press = pressure_events(kin, poss, {1: "home", 2: "away"}, FPS, PressingConfig())
    assert len(press) >= 1
    assert (press.presser_team == "away").all()
    assert (press.target_id == 1).all()


def test_ppda_lower_for_pressing_team():
    from pitchiq.analytics.events import EVENT_COLUMNS

    meta = _meta(1000)
    # away makes many build-up passes; home logs many pressures in that zone
    passes = [dict(frame=f, timestamp=f / FPS, type="pass", team="away",
                   from_id=101, to_id=102, x=80.0, y=30.0, end_x=70.0, end_y=30.0,
                   outcome="complete") for f in range(0, 500, 25)]
    events = pd.DataFrame(passes, columns=EVENT_COLUMNS)
    press = pd.DataFrame([
        dict(frame=f, presser_id=5, presser_team="home", target_id=101,
             dist_m=2.0, closing_mps=3.0, x=80.0, y=30.0, duration_frames=5)
        for f in range(0, 500, 50)
    ])
    result = ppda(events, press, meta, PressingConfig())
    assert result["home"]["ppda"] == 2.0  # 20 passes / 10 pressures
    assert result["home"]["opp_buildup_passes"] == 20


def test_shape_line_height_and_tilt():
    rows = []
    for f in range(0, 300):
        # home: back four at x=30, mids at 45, forwards at 60 (attacks +x)
        for i, x in enumerate([30, 30, 30, 30, 45, 45, 45, 60, 60, 60]):
            rows.append(_row(f, 1 + i, "player", "home", x, 6 + 6 * i))
        for i, x in enumerate([70, 70, 70, 70, 55, 55, 55, 40, 40, 40]):
            rows.append(_row(f, 101 + i, "player", "away", x, 6 + 6 * i))
        rows.append(_row(f, -1, "ball", "none", 80.0, 34.0))  # deep in away half
    df = pd.DataFrame(rows)
    poss = compute_possession(df, FPS, PossessionConfig(control_radius_m=30))
    meta = _meta(300)
    ts = shape_timeseries(df, poss, meta)
    home = ts[ts.team == "home"]
    assert abs(home.def_line_height_m.mean() - 30.0) < 2.0
    assert abs(home.depth_m.mean() - 30.0) < 2.0
    summary = shape_summary(ts)
    assert summary["home"]
    tilt = field_tilt(df, poss, meta)
    # nearest to the ball at (80,34) is an away defender -> away possession;
    # away attacks -x, so x=80 is deep in AWAY's own third
    assert tilt["away"]["own_third_share"] > 0.9
    assert "tilt_home" not in tilt or 0 <= tilt["tilt_home"] <= 1
