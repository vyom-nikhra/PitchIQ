"""Unit tests for possession, events, kinematics on constructed micro-fixtures."""

import numpy as np
import pandas as pd
import pytest

from pitchiq.config import KinematicsConfig, PassesConfig, PossessionConfig
from pitchiq.analytics.events import derive_events
from pitchiq.analytics.kinematics import compute_kinematics, player_movement_profiles
from pitchiq.analytics.possession import compute_possession, possession_summary

FPS = 25.0


def _row(frame, eid, cls, team, x, y):
    return dict(frame=frame, timestamp=frame / FPS, entity_id=eid, **{"class": cls},
                team=team, jersey_no=None, x_pixel=np.nan, y_pixel=np.nan,
                x_pitch=x, y_pitch=y, conf=1.0)


def _pass_chain_df(n_frames=200):
    """P1 (home) holds the ball, passes to P2 at f=50 (arrives f=75); P2 passes
    to away P3 at f=140 (interception scenario arrives f=160)."""
    rows = []
    p1 = (30.0, 30.0)
    p2 = (50.0, 40.0)
    p3 = (60.0, 20.0)
    for f in range(n_frames):
        rows.append(_row(f, 1, "player", "home", *p1))
        rows.append(_row(f, 2, "player", "home", *p2))
        rows.append(_row(f, 3, "player", "away", *p3))
        if f < 50:
            bx, by = p1
        elif f < 75:
            t = (f - 50) / 25
            bx = p1[0] + (p2[0] - p1[0]) * t
            by = p1[1] + (p2[1] - p1[1]) * t
        elif f < 140:
            bx, by = p2
        elif f < 160:
            t = (f - 140) / 20
            bx = p2[0] + (p3[0] - p2[0]) * t
            by = p2[1] + (p3[1] - p2[1]) * t
        else:
            bx, by = p3
        rows.append(_row(f, -1, "ball", "none", bx, by))
    return pd.DataFrame(rows)


def test_possession_switches_with_hysteresis():
    df = _pass_chain_df()
    poss = compute_possession(df, FPS, PossessionConfig())
    assert (poss[poss.frame < 45].holder_id == 1).all()
    late = poss[(poss.frame > 90) & (poss.frame < 130)]
    assert (late.holder_id == 2).all()
    assert (poss[poss.frame > 180].holder_id == 3).all()
    # no flicker: holder changes exactly twice
    changes = (poss.holder_id != poss.holder_id.shift()).sum() - 1
    assert changes == 2
    summary = possession_summary(poss, FPS)
    assert summary["share"]["home"] > 0.6


def test_derive_events_pass_and_turnover():
    df = _pass_chain_df()
    poss = compute_possession(df, FPS, PossessionConfig())
    ev = derive_events(df, poss, FPS, PassesConfig())
    passes = ev[ev.type == "pass"]
    assert len(passes) == 2
    complete = passes[passes.outcome == "complete"]
    assert len(complete) == 1
    p = complete.iloc[0]
    assert (p.from_id, p.to_id) == (1, 2)
    # origin near P1, end near P2
    assert np.hypot(p.x - 30, p.y - 30) < 3.0
    assert np.hypot(p.end_x - 50, p.end_y - 40) < 3.0
    turnovers = ev[ev.type == "turnover"]
    assert len(turnovers) == 1
    assert turnovers.iloc[0].team == "away"


def test_kinematics_constant_velocity():
    rows = []
    v = 4.0  # m/s along +x
    for f in range(150):
        rows.append(_row(f, 1, "player", "home", 10 + v * f / FPS, 30.0))
        rows.append(_row(f, -1, "ball", "none", 50, 34))
    df = pd.DataFrame(rows)
    kin = compute_kinematics(df, FPS, KinematicsConfig())
    mid = kin[(kin.frame > 20) & (kin.frame < 130)]
    assert mid.speed.mean() == pytest.approx(v, abs=0.2)
    prof = player_movement_profiles(kin, FPS, KinematicsConfig())[1]
    expected_dist = v * 150 / FPS
    assert prof["distance_m"] == pytest.approx(expected_dist, rel=0.1)
    assert prof["n_sprints"] == 0


def test_kinematics_sprint_detection():
    rows = []
    for f in range(200):
        speed = 8.0 if 80 <= f < 120 else 2.0
        x = 5 + sum(8.0 / FPS if 80 <= i < 120 else 2.0 / FPS for i in range(f))
        rows.append(_row(f, 1, "player", "home", min(x, 100), 30.0))
        rows.append(_row(f, -1, "ball", "none", 50, 34))
    df = pd.DataFrame(rows)
    kin = compute_kinematics(df, FPS, KinematicsConfig())
    prof = player_movement_profiles(kin, FPS, KinematicsConfig())[1]
    assert prof["n_sprints"] == 1
    assert prof["top_speed_mps"] > 7.0
    assert prof["hi_distance_m"] > 10
