"""Space control: Voronoi tessellation and a velocity-aware pitch-control model.

Two granularities:

* :func:`voronoi_areas` — classic Voronoi cell areas per team (fast, purely
  positional). Cells are clipped to the pitch by mirroring players across the
  boundaries (standard finite-cell construction).
* :func:`control_grid` — a simplified Spearman-style model: each player's
  time-to-reach every grid cell is reaction time + distance from their
  reaction-rolled position at max speed; team control is a logistic of the
  best arrival-time difference. This is the per-frame surface behind the
  space-control visuals and off-ball-run valuation. (Full Spearman integrates
  ball time-of-flight and control duration; documented simplification.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Voronoi

from pitchiq.config import PitchControlConfig


def _grid(cfg: PitchControlConfig, length: float, width: float):
    xs = np.linspace(0, length, cfg.grid_nx)
    ys = np.linspace(0, width, cfg.grid_ny)
    return np.meshgrid(xs, ys)  # (ny, nx)


def control_grid(
    positions: np.ndarray,
    velocities: np.ndarray,
    teams: np.ndarray,
    cfg: PitchControlConfig,
    length: float = 105.0,
    width: float = 68.0,
) -> np.ndarray:
    """P(home controls cell) over the grid for one frame.

    ``positions``/``velocities``: (N,2); ``teams``: array of 'home'/'away'.
    Returns (ny, nx) float array in [0,1].
    """
    gx, gy = _grid(cfg, length, width)
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (M,2)
    pos = np.asarray(positions, dtype=float)
    vel = np.nan_to_num(np.asarray(velocities, dtype=float))
    # position after the reaction time, drifting on current velocity
    rolled = pos + vel * cfg.reaction_time_s
    d = np.linalg.norm(rolled[:, None, :] - cells[None, :, :], axis=2)  # (N,M)
    tti = cfg.reaction_time_s + d / cfg.max_speed_mps
    is_home = np.asarray(teams) == "home"
    if not is_home.any() or is_home.all():
        return np.full(gx.shape, 0.5, dtype=np.float32)
    t_home = tti[is_home].min(axis=0)
    t_away = tti[~is_home].min(axis=0)
    p_home = 1.0 / (1.0 + np.exp((t_home - t_away) / cfg.kappa))
    return p_home.reshape(gx.shape).astype(np.float32)


def mean_control(
    kin: pd.DataFrame,
    team_of: dict[int, str],
    cfg: PitchControlConfig,
    length: float = 105.0,
    width: float = 68.0,
    every_n: int = 12,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Average control surface + per-frame team control shares (sampled)."""
    frames = sorted(kin["frame"].unique())[::every_n]
    acc = np.zeros((cfg.grid_ny, cfg.grid_nx), dtype=np.float64)
    rows = []
    n = 0
    by_frame = dict(tuple(kin.groupby("frame")))
    for f in frames:
        g = by_frame.get(f)
        if g is None:
            continue
        teams = g["entity_id"].map(team_of).to_numpy(dtype=object)
        keep = (teams == "home") | (teams == "away")
        if keep.sum() < 6:
            continue
        grid = control_grid(
            g[["x", "y"]].to_numpy()[keep], g[["vx", "vy"]].to_numpy()[keep],
            teams[keep], cfg, length, width,
        )
        acc += grid
        n += 1
        third = np.array_split(grid, 3, axis=1)
        rows.append(dict(
            frame=f,
            home_control=float(grid.mean()),
            home_control_final_third_right=float(third[2].mean()),
            home_control_final_third_left=float(third[0].mean()),
        ))
    mean_grid = (acc / max(n, 1)).astype(np.float32)
    return mean_grid, pd.DataFrame(rows)


def voronoi_areas(
    positions: np.ndarray, teams: np.ndarray, length: float = 105.0, width: float = 68.0
) -> dict[str, float]:
    """Voronoi-controlled area (m²) per team for one frame.

    Finite cells via the mirror trick: reflect every player across all four
    pitch edges; interior cells then clip exactly to the pitch rectangle.
    """
    pos = np.asarray(positions, dtype=float)
    if len(pos) < 4:
        return {"home": np.nan, "away": np.nan}
    mirrored = [pos]
    for axis, bound in ((0, 0.0), (0, length), (1, 0.0), (1, width)):
        m = pos.copy()
        m[:, axis] = 2 * bound - m[:, axis]
        mirrored.append(m)
    allp = np.concatenate(mirrored)
    vor = Voronoi(allp)
    areas = {"home": 0.0, "away": 0.0}
    for i in range(len(pos)):
        region = vor.regions[vor.point_region[i]]
        if -1 in region or len(region) < 3:
            continue
        poly = vor.vertices[region]
        try:
            area = float(ConvexHull(poly).volume)
        except Exception:
            continue
        t = str(teams[i])
        if t in areas:
            areas[t] += area
    return {k: round(v, 1) for k, v in areas.items()}
