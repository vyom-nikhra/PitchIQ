"""Formation detection & transitions via Hungarian template matching.

For each analysis window we take the team's 10 most-present outfielders, mean
their attacking-normalised positions, scale-normalise, and match against the
canonical templates in :mod:`pitchiq.core.formations` with the Hungarian
algorithm (min total squared slot distance). Windows are split by possession
state so the in-possession vs out-of-possession shape morph (e.g. 4-3-3 →
4-5-1) falls out naturally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from pitchiq.config import FormationsConfig
from pitchiq.core.formations import FORMATIONS, formation_labels
from pitchiq.core.schema import MatchMeta
from pitchiq.analytics.common import attack_sign_series, to_attacking_xy


def match_formation(mean_positions: np.ndarray) -> tuple[str, float, list[str]]:
    """Match 10 attacking-normalised mean positions to the closest template.

    Returns (formation_name, cost, slot_label_per_player). Positions and
    templates are centred and scaled to unit std so the match measures shape,
    not pitch location.
    """
    P = np.asarray(mean_positions, dtype=float)
    if P.shape != (10, 2):
        raise ValueError(f"need exactly 10 outfield positions, got {P.shape}")

    def normalise(a: np.ndarray) -> np.ndarray:
        a = a - a.mean(axis=0)
        s = a.std()
        return a / s if s > 1e-9 else a

    Pn = normalise(P)
    best = ("unknown", np.inf, [""] * 10)
    for name, slots in FORMATIONS.items():
        T = normalise(np.array([(x, y) for x, y, _ in slots], dtype=float))
        cost = ((Pn[:, None, :] - T[None, :, :]) ** 2).sum(axis=2)
        rows, cols = linear_sum_assignment(cost)
        total = float(cost[rows, cols].sum())
        if total < best[1]:
            labels = formation_labels(name)
            per_player = [""] * 10
            for r, c in zip(rows, cols):
                per_player[r] = labels[c]
            best = (name, total, per_player)
    return best


def formation_windows(
    df: pd.DataFrame,
    possession: pd.DataFrame,
    meta: MatchMeta,
    cfg: FormationsConfig,
) -> pd.DataFrame:
    """Formation per (team, possession-state) window.

    Columns: team, state (in/out), start_frame, end_frame, formation, cost,
    hull_area_m2, width_m, depth_m, players (ids), slots (labels).
    """
    fps = meta.fps
    win = int(cfg.window_s * fps)
    outfield = df[(df["class"] == "player") & df["team"].isin(["home", "away"])].dropna(
        subset=["x_pitch", "y_pitch"]
    )
    poss_map = possession.set_index("frame")["team"]
    rows = []
    max_frame = int(df["frame"].max())
    for start in range(0, max_frame + 1, win):
        end = min(start + win - 1, max_frame)
        chunk = outfield[(outfield.frame >= start) & (outfield.frame <= end)]
        poss_chunk = poss_map.reindex(range(start, end + 1)).dropna()
        for team in ("home", "away"):
            g = chunk[chunk.team == team]
            if g.frame.nunique() < win * 0.5:
                continue
            for state in ("in", "out"):
                state_frames = poss_chunk[poss_chunk == team].index if state == "in" \
                    else poss_chunk[(poss_chunk != team) & (poss_chunk != "none")].index
                sg = g[g.frame.isin(state_frames)]
                # need enough observation to define a shape
                if sg.frame.nunique() < fps * 3:
                    continue
                top = sg.groupby("entity_id").size().nlargest(10)
                if len(top) < cfg.min_window_players:
                    continue
                ids = top.index.tolist()[:10]
                if len(ids) < 10:
                    continue
                means = sg[sg.entity_id.isin(ids)].groupby("entity_id")[
                    ["x_pitch", "y_pitch"]].mean().loc[ids].to_numpy()
                signs = attack_sign_series(meta, np.array([start]), team)
                means_att = to_attacking_xy(means, np.repeat(signs, len(means)),
                                            meta.pitch_length, meta.pitch_width)
                name, cost, slots = match_formation(means_att)
                hull_area = _hull_area(means)
                rows.append(dict(
                    team=team, state=state, start_frame=start, end_frame=end,
                    formation=name, cost=round(cost, 3),
                    hull_area_m2=round(hull_area, 1),
                    depth_m=round(float(means_att[:, 0].max() - means_att[:, 0].min()), 1),
                    width_m=round(float(means_att[:, 1].max() - means_att[:, 1].min()), 1),
                    players=[int(i) for i in ids], slots=slots,
                ))
    return pd.DataFrame(rows)


def _hull_area(points: np.ndarray) -> float:
    from scipy.spatial import ConvexHull

    if len(points) < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume)  # 2D: volume == area
    except Exception:
        return 0.0


def formation_summary(windows: pd.DataFrame) -> dict:
    """Modal formation per team/state + morph description + transitions."""
    out: dict = {}
    for team, g in windows.groupby("team"):
        team_sum: dict = {}
        for state, sg in g.groupby("state"):
            if not len(sg):
                continue
            modal = sg["formation"].mode().iat[0]
            team_sum[f"{state}_possession"] = {
                "formation": modal,
                "stability": round(float((sg["formation"] == modal).mean()), 2),
                "avg_hull_area_m2": round(float(sg["hull_area_m2"].mean()), 1),
                "avg_width_m": round(float(sg["width_m"].mean()), 1),
                "avg_depth_m": round(float(sg["depth_m"].mean()), 1),
            }
        # transition sequence over time (both states interleaved by window)
        seq = g.sort_values(["start_frame", "state"])[["state", "formation"]].to_records(index=False)
        transitions = []
        prev = {}
        for state, form in seq:
            if state in prev and prev[state] != form:
                transitions.append({"state": state, "from": prev[state], "to": form})
            prev[state] = form
        team_sum["transitions"] = transitions
        morph_in = team_sum.get("in_possession", {}).get("formation")
        morph_out = team_sum.get("out_possession", {}).get("formation")
        if morph_in and morph_out:
            team_sum["shape_morph"] = f"{morph_in} in possession → {morph_out} out of possession"
        out[str(team)] = team_sum
    return out
