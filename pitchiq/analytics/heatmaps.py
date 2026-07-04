"""Positional heatmaps (players & teams) + territory occupation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from pitchiq.config import HeatmapConfig
from pitchiq.analytics.common import players_only


def position_heatmap(x: np.ndarray, y: np.ndarray, cfg: HeatmapConfig,
                     length: float = 105.0, width: float = 68.0) -> np.ndarray:
    """(ny, nx) smoothed, sum-normalised occupancy grid (row 0 = y=0 edge)."""
    hist, _, _ = np.histogram2d(
        y, x, bins=[cfg.ny, cfg.nx], range=[[0, width], [0, length]]
    )
    hist = gaussian_filter(hist, sigma=cfg.sigma_cells)
    total = hist.sum()
    return (hist / total if total > 0 else hist).astype(np.float32)


def all_heatmaps(df: pd.DataFrame, cfg: HeatmapConfig,
                 length: float = 105.0, width: float = 68.0) -> dict[str, np.ndarray]:
    """'team_home', 'team_away' and 'player_<id>' heatmaps."""
    persons = players_only(df)
    out: dict[str, np.ndarray] = {}
    for team, g in persons.groupby("team"):
        out[f"team_{team}"] = position_heatmap(
            g.x_pitch.to_numpy(), g.y_pitch.to_numpy(), cfg, length, width)
    for eid, g in persons.groupby("entity_id"):
        out[f"player_{int(eid)}"] = position_heatmap(
            g.x_pitch.to_numpy(), g.y_pitch.to_numpy(), cfg, length, width)
    return out


def third_occupation(df: pd.DataFrame, length: float = 105.0) -> dict:
    """Share of each team's player-observations per pitch third (raw x)."""
    persons = players_only(df)
    thirds = np.digitize(persons.x_pitch, [length / 3, 2 * length / 3])
    persons = persons.assign(third=thirds)
    out = {}
    for team, g in persons.groupby("team"):
        counts = g["third"].value_counts(normalize=True)
        out[str(team)] = {
            "defensive_third_left": round(float(counts.get(0, 0)), 3),
            "middle_third": round(float(counts.get(1, 0)), 3),
            "final_third_right": round(float(counts.get(2, 0)), 3),
        }
    return out
