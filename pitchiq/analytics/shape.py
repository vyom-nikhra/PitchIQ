"""Team shape: defensive line height, compactness, field tilt."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitchiq.core.schema import MatchMeta
from pitchiq.analytics.common import attack_sign_series, ball_series, to_attacking_x


def shape_timeseries(df: pd.DataFrame, possession: pd.DataFrame, meta: MatchMeta,
                     every_n: int = 5) -> pd.DataFrame:
    """Per sampled frame & team: defensive line height, compactness, hull.

    ``def_line_height_m`` — distance of the second-deepest outfielder from
    their own goal line (the deepest is usually the GK-adjacent sweeper;
    second-deepest is the standard line proxy) measured in the team's own
    defensive frame.
    """
    outfield = df[(df["class"] == "player") & df["team"].isin(["home", "away"])].dropna(
        subset=["x_pitch", "y_pitch"])
    poss_map = possession.set_index("frame")["team"]
    rows = []
    for f, g in outfield.groupby("frame"):
        if f % every_n:
            continue
        for team, tg in g.groupby("team"):
            if len(tg) < 7:
                continue
            sign = attack_sign_series(meta, np.array([f]), str(team))[0]
            x_att = to_attacking_x(tg.x_pitch.to_numpy(), np.full(len(tg), sign),
                                   meta.pitch_length)
            x_sorted = np.sort(x_att)
            rows.append(dict(
                frame=f, team=team,
                def_line_height_m=round(float(x_sorted[1]), 2),
                depth_m=round(float(x_att.max() - x_att.min()), 2),
                width_m=round(float(tg.y_pitch.max() - tg.y_pitch.min()), 2),
                centroid_x_att=round(float(x_att.mean()), 2),
                in_possession=bool(poss_map.get(f) == team),
            ))
    return pd.DataFrame(rows)


def shape_summary(ts: pd.DataFrame) -> dict:
    out = {}
    for team, g in ts.groupby("team"):
        team_sum = {}
        for state, sg in (("in_possession", g[g.in_possession]),
                          ("out_possession", g[~g.in_possession])):
            if not len(sg):
                continue
            team_sum[state] = {
                "def_line_height_m": round(float(sg.def_line_height_m.mean()), 1),
                "depth_m": round(float(sg.depth_m.mean()), 1),
                "width_m": round(float(sg.width_m.mean()), 1),
            }
        lh = team_sum.get("out_possession", {}).get("def_line_height_m", 35.0)
        team_sum["block_label"] = ("high line" if lh > 40 else
                                   "mid block" if lh > 28 else "deep block")
        out[str(team)] = team_sum
    return out


def field_tilt(df: pd.DataFrame, possession: pd.DataFrame, meta: MatchMeta) -> dict:
    """Field tilt: share of in-possession ball time spent in the FINAL third,
    per team (the standard 'who pushes whom back' metric)."""
    ball = ball_series(df).reset_index()
    poss = possession.set_index("frame")["team"]
    ball["team"] = ball["frame"].map(poss)
    ball = ball.dropna(subset=["x_pitch"])
    ball = ball[ball["team"].isin(["home", "away"])]
    out = {}
    for team, g in ball.groupby("team"):
        signs = attack_sign_series(meta, g["frame"].to_numpy(), str(team))
        x_att = to_attacking_x(g["x_pitch"].to_numpy(), signs, meta.pitch_length)
        out[str(team)] = {
            "final_third_share": round(float((x_att > 2 * meta.pitch_length / 3).mean()), 3),
            "own_third_share": round(float((x_att < meta.pitch_length / 3).mean()), 3),
            "mean_ball_x_att": round(float(x_att.mean()), 1),
        }
    # tilt = A's final-third time / (A's + B's final-third time)
    ft_h = out.get("home", {}).get("final_third_share", 0)
    ft_a = out.get("away", {}).get("final_third_share", 0)
    if ft_h + ft_a > 0:
        out["tilt_home"] = round(ft_h / (ft_h + ft_a), 3)
    return out
