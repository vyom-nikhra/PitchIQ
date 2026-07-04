"""Expected Threat (xT): grid value model fitted by value iteration.

Karun Singh's formulation: split the pitch into a grid (attacking frame);
from observed events estimate, per cell, the shot probability, goal-given-
shot probability, and the move transition matrix; then iterate

    xT(s) = P(shot|s)·P(goal|shot,s) + P(move|s)·Σ_s' P(s'|s)·xT(s')

until convergence. Completed moves (passes/carries) credit the mover with
ΔxT = xT(end) − xT(start).

With a short clip the transition statistics are sparse — Laplace smoothing +
a distance-to-goal shot prior keep the surface sane; ``grid_path`` can point
to a pre-fitted grid (e.g. from StatsBomb open events) to skip fitting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitchiq.config import XTConfig
from pitchiq.core.schema import MatchMeta
from pitchiq.analytics.common import attack_sign_series, to_attacking_xy


def _cell_of(x_att: np.ndarray, y: np.ndarray, cfg: XTConfig,
             length: float, width: float) -> np.ndarray:
    cx = np.clip((x_att / length * cfg.grid_nx).astype(int), 0, cfg.grid_nx - 1)
    cy = np.clip((y / width * cfg.grid_ny).astype(int), 0, cfg.grid_ny - 1)
    return cy * cfg.grid_nx + cx


def fit_xt_grid(events: pd.DataFrame, meta: MatchMeta, cfg: XTConfig) -> np.ndarray:
    """Fit the xT surface from this match's events (attacking frame, ny x nx)."""
    n_cells = cfg.grid_nx * cfg.grid_ny
    L, W = meta.pitch_length, meta.pitch_width

    moves = events[(events.type.isin(["pass", "carry"])) & (events.outcome == "complete")].copy()
    shots = events[events.type == "shot"] if "shot" in set(events.type) else pd.DataFrame()

    # transition prior: moves decay with distance and skew slightly forward —
    # a uniform Laplace prior would let value "teleport" and flatten the
    # surface when event data is sparse (a few minutes of tracking)
    gx0, gy0 = np.meshgrid(np.arange(cfg.grid_nx), np.arange(cfg.grid_ny))
    cx_m = (gx0.ravel() + 0.5) / cfg.grid_nx * L
    cy_m = (gy0.ravel() + 0.5) / cfg.grid_ny * W
    dmat = np.hypot(cx_m[:, None] - cx_m[None, :], cy_m[:, None] - cy_m[None, :])
    forward = np.clip(cx_m[None, :] - cx_m[:, None], -30, 30)
    move_prior = np.exp(-dmat / 18.0) * np.exp(forward / 45.0)
    np.fill_diagonal(move_prior, 0.0)
    move_prior /= move_prior.sum(axis=1, keepdims=True)

    move_counts = move_prior * 2.0  # prior weight ≈ 2 observed moves per cell
    shot_counts = np.zeros(n_cells)
    goal_counts = np.zeros(n_cells)
    total_counts = np.full(n_cells, 1e-6)

    if len(moves):
        signs = attack_sign_series(meta, moves.frame.to_numpy(), "home")
        team_signs = np.where(moves.team.to_numpy() == "home", signs, -signs)
        start = to_attacking_xy(moves[["x", "y"]].to_numpy(), team_signs, L, W)
        end = to_attacking_xy(moves[["end_x", "end_y"]].to_numpy(), team_signs, L, W)
        s_cells = _cell_of(start[:, 0], start[:, 1], cfg, L, W)
        e_cells = _cell_of(end[:, 0], end[:, 1], cfg, L, W)
        for s, e in zip(s_cells, e_cells):
            move_counts[s, e] += 1
            total_counts[s] += 1
    if len(shots):
        signs = attack_sign_series(meta, shots.frame.to_numpy(), "home")
        team_signs = np.where(shots.team.to_numpy() == "home", signs, -signs)
        pos = to_attacking_xy(shots[["x", "y"]].to_numpy(), team_signs, L, W)
        cells = _cell_of(pos[:, 0], pos[:, 1], cfg, L, W)
        for c, outcome in zip(cells, shots.outcome):
            shot_counts[c] += 1
            total_counts[c] += 1
            if outcome == "goal":
                goal_counts[c] += 1

    # priors keep sparse cells sensible: shot tendency and conversion both
    # rise with proximity to goal (goal at x=L, y=W/2 in the attacking frame)
    gx, gy = np.meshgrid(np.arange(cfg.grid_nx), np.arange(cfg.grid_ny))
    cell_x = (gx.ravel() + 0.5) / cfg.grid_nx * L
    cell_y = (gy.ravel() + 0.5) / cfg.grid_ny * W
    d_goal = np.hypot(L - cell_x, W / 2 - cell_y)
    shot_prior = np.clip(0.9 - d_goal / 40.0, 0.01, 0.85)
    goal_prior = np.clip(0.55 - d_goal / 45.0, 0.02, 0.5)

    prior_w = 8.0
    p_shot = (shot_counts + prior_w * shot_prior * 0.15) / (total_counts + prior_w)
    p_goal = (goal_counts + 3.0 * goal_prior) / (shot_counts + 3.0)
    p_move = 1.0 - p_shot
    T = move_counts / move_counts.sum(axis=1, keepdims=True)

    # moves fail (interceptions/out): without this damping, value telegraphs
    # across the pitch through long chains and the surface goes flat
    all_passes = events[events.type == "pass"]
    if len(all_passes) >= 20:
        p_complete = float((all_passes.outcome == "complete").mean())
        p_complete = float(np.clip(p_complete, 0.5, 0.95))
    else:
        p_complete = 0.78

    xt = np.zeros(n_cells)
    for _ in range(cfg.n_iterations):
        xt = p_shot * p_goal + p_move * p_complete * (T @ xt)
    return xt.reshape(cfg.grid_ny, cfg.grid_nx).astype(np.float32)


def load_or_fit_grid(events: pd.DataFrame, meta: MatchMeta, cfg: XTConfig) -> np.ndarray:
    if cfg.grid_path and Path(cfg.grid_path).exists():
        grid = np.array(json.loads(Path(cfg.grid_path).read_text()), dtype=np.float32)
        if grid.shape == (cfg.grid_ny, cfg.grid_nx):
            return grid
    return fit_xt_grid(events, meta, cfg)


def player_xt_contributions(events: pd.DataFrame, grid: np.ndarray, meta: MatchMeta,
                            cfg: XTConfig) -> pd.DataFrame:
    """Sum of ΔxT over each player's completed moves (positive moves only
    counted toward 'threat created'; net also reported)."""
    L, W = meta.pitch_length, meta.pitch_width
    moves = events[(events.type.isin(["pass", "carry"])) & (events.outcome == "complete")].copy()
    if not len(moves):
        return pd.DataFrame(columns=["entity_id", "team", "xt_net", "xt_created", "n_moves"])
    signs = attack_sign_series(meta, moves.frame.to_numpy(), "home")
    team_signs = np.where(moves.team.to_numpy() == "home", signs, -signs)
    start = to_attacking_xy(moves[["x", "y"]].to_numpy(), team_signs, L, W)
    end = to_attacking_xy(moves[["end_x", "end_y"]].to_numpy(), team_signs, L, W)
    flat = grid.ravel()
    delta = (flat[_cell_of(end[:, 0], end[:, 1], cfg, L, W)]
             - flat[_cell_of(start[:, 0], start[:, 1], cfg, L, W)])
    moves["dxt"] = delta
    agg = moves.groupby(["from_id", "team"]).agg(
        xt_net=("dxt", "sum"),
        xt_created=("dxt", lambda s: float(s[s > 0].sum())),
        n_moves=("dxt", "size"),
    ).reset_index().rename(columns={"from_id": "entity_id"})
    agg["xt_net"] = agg["xt_net"].round(4)
    agg["xt_created"] = agg["xt_created"].round(4)
    return agg.sort_values("xt_created", ascending=False)
