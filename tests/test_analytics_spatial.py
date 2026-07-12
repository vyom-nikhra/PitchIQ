"""Tests for formations, pitch control, Voronoi, heatmaps, shape, xT."""

import numpy as np
import pandas as pd
import pytest

from pitchiq.config import HeatmapConfig, PitchControlConfig, XTConfig
from pitchiq.core.formations import formation_slots
from pitchiq.core.schema import MatchMeta
from pitchiq.analytics.formations import match_formation
from pitchiq.analytics.heatmaps import position_heatmap
from pitchiq.analytics.pitch_control import control_grid, voronoi_areas
from pitchiq.analytics.xt import fit_xt_grid, load_or_fit_grid


def test_match_formation_recovers_templates():
    rng = np.random.default_rng(0)
    for name in ("4-4-2", "4-3-3", "3-5-2"):
        slots = formation_slots(name)
        noisy = slots + rng.normal(0, 1.2, slots.shape)
        detected, cost, labels = match_formation(noisy)
        assert detected == name, f"{name} detected as {detected}"
        assert len(labels) == 10


def test_control_grid_symmetry_and_dominance():
    cfg = PitchControlConfig()
    pos = np.array([[40.0, 34.0], [65.0, 34.0]])
    vel = np.zeros((2, 2))
    teams = np.array(["home", "away"])
    grid = control_grid(pos, vel, teams, cfg)
    assert grid.shape == (cfg.grid_ny, cfg.grid_nx)
    assert np.all((grid >= 0) & (grid <= 1))
    # midpoint between the two is contested
    mid_val = grid[cfg.grid_ny // 2, cfg.grid_nx // 2]
    assert 0.35 < mid_val < 0.65
    # cell on home player dominated by home
    home_cell = grid[cfg.grid_ny // 2, int(40 / 105 * cfg.grid_nx)]
    assert home_cell > 0.9


def test_control_grid_velocity_matters():
    cfg = PitchControlConfig()
    pos = np.array([[50.0, 34.0], [55.0, 34.0]])
    teams = np.array(["home", "away"])
    still = control_grid(pos, np.zeros((2, 2)), teams, cfg)
    # home sprinting right shifts control right of centre toward home
    running = control_grid(pos, np.array([[7.0, 0.0], [0.0, 0.0]]), teams, cfg)
    x_probe = int(60 / 105 * cfg.grid_nx)
    assert running[cfg.grid_ny // 2, x_probe] > still[cfg.grid_ny // 2, x_probe]


def test_voronoi_symmetric_split():
    pos = np.array([[30, 20], [30, 48], [75, 20], [75, 48]], dtype=float)
    teams = np.array(["home", "home", "away", "away"])
    areas = voronoi_areas(pos, teams)
    total = areas["home"] + areas["away"]
    assert total == pytest.approx(105 * 68, rel=0.02)
    assert areas["home"] == pytest.approx(areas["away"], rel=0.05)


def test_impute_offscreen_ghosts_decay_and_expire():
    """A player who leaves frame persists as a drifting ghost for the horizon
    (velocity settles), then disappears; observed rows are never altered."""
    from pitchiq.analytics.pitch_control import impute_offscreen

    fps = 25.0
    rows = []
    for f in range(10):  # visible frames 0-9, running +5 m/s in x
        rows.append(dict(frame=f, entity_id=1, x=20.0 + 0.2 * f, y=30.0,
                         vx=5.0, vy=0.0))
        rows.append(dict(frame=f, entity_id=2, x=80.0, y=40.0, vx=0.0, vy=0.0))
    for f in range(10, 60):  # player 1 leaves; player 2 stays
        rows.append(dict(frame=f, entity_id=2, x=80.0, y=40.0, vx=0.0, vy=0.0))
    kin = pd.DataFrame(rows)

    out = impute_offscreen(kin, horizon_s=1.0, fps=fps)
    g1 = out[(out.entity_id == 1)]
    ghosts = g1[g1.ghost]
    assert len(ghosts) == 25                      # exactly 1 s of ghosts
    assert ghosts.frame.max() == 34               # then we stop pretending
    # drift: moved forward from last observed x, but less than full speed
    last_x = 20.0 + 0.2 * 9
    assert ghosts.x.iloc[-1] > last_x + 1.0
    assert ghosts.x.iloc[-1] < last_x + 5.0 * 1.0
    # velocity decays toward zero
    assert abs(ghosts.vx.iloc[-1]) < 5.0 * 0.6
    # observed rows untouched, always-visible player gains no ghosts
    assert not out[(out.entity_id == 2) & out.ghost].shape[0]
    assert len(g1[~g1.ghost]) == 10


def test_heatmap_normalised_and_peaked():
    cfg = HeatmapConfig()
    x = np.random.default_rng(1).normal(80, 3, 500).clip(0, 104)
    y = np.random.default_rng(2).normal(20, 3, 500).clip(0, 67)
    hm = position_heatmap(x, y, cfg)
    assert hm.sum() == pytest.approx(1.0, abs=1e-5)
    iy, ix = np.unravel_index(hm.argmax(), hm.shape)
    assert abs(ix / cfg.nx * 105 - 80) < 8
    assert abs(iy / cfg.ny * 68 - 20) < 8


def _empty_events():
    from pitchiq.analytics.events import EVENT_COLUMNS

    return pd.DataFrame(columns=EVENT_COLUMNS)


def test_xt_grid_monotone_toward_goal():
    meta = MatchMeta(fps=25, n_frames=100)
    grid = fit_xt_grid(_empty_events(), meta, XTConfig())
    ny, nx = grid.shape
    row = grid[ny // 2]
    # threat strictly higher in the attacking (right) sixth than own sixth
    assert row[-2] > row[1] * 3
    assert np.all(grid >= 0) and np.all(grid <= 1)


def test_xt_grid_load_path(tmp_path):
    import json

    meta = MatchMeta(fps=25, n_frames=100)
    cfg = XTConfig()
    ref = np.ones((cfg.grid_ny, cfg.grid_nx)) * 0.05
    p = tmp_path / "grid.json"
    p.write_text(json.dumps(ref.tolist()))
    cfg2 = XTConfig(grid_path=str(p))
    grid = load_or_fit_grid(_empty_events(), meta, cfg2)
    assert np.allclose(grid, 0.05)
