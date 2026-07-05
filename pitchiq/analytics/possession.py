"""Ball possession from tracking: nearest-player with hysteresis.

The naive nearest-player signal flickers whenever players cross the ball's
path; hysteresis requires a *different* candidate to persist for
``hysteresis_frames`` consecutive frames before possession transfers, and a
candidate only counts when within ``control_radius_m`` of the ball.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitchiq.config import PossessionConfig
from pitchiq.analytics.common import ball_series, players_only


def compute_possession(df: pd.DataFrame, fps: float, cfg: PossessionConfig) -> pd.DataFrame:
    """Per-frame possession: columns frame, holder_id, team, dist_m.

    holder_id = -1 / team='none' where nobody controls the ball (in flight,
    dead, or no ball observation).

    Frames whose ball row lacks pitch coordinates (uncalibrated frames in CV
    output) are excluded up front — they would otherwise form all-NaN
    distance groups, which pandas' grouped idxmin refuses.
    """
    ball = ball_series(df).dropna(subset=["x_pitch", "y_pitch"])
    persons = players_only(df)
    merged = persons.merge(
        ball[["x_pitch", "y_pitch"]].rename(columns={"x_pitch": "bx", "y_pitch": "by"}),
        left_on="frame", right_index=True, how="inner",
    )
    merged["dist"] = np.hypot(merged.x_pitch - merged.bx, merged.y_pitch - merged.by)
    merged = merged.dropna(subset=["dist"])
    if merged.empty:
        return pd.DataFrame(columns=["frame", "holder_id", "team", "dist_m"])
    nearest = merged.loc[merged.groupby("frame")["dist"].idxmin(),
                         ["frame", "entity_id", "team", "dist"]]

    frames = nearest["frame"].to_numpy()
    cand_ids = nearest["entity_id"].to_numpy()
    cand_team = nearest["team"].to_numpy(dtype=object)
    cand_dist = nearest["dist"].to_numpy()

    holder = np.full(len(frames), -1, dtype=int)
    team = np.full(len(frames), "none", dtype=object)
    cur_holder, cur_team = -1, "none"
    challenger, challenge_len = -1, 0
    for i in range(len(frames)):
        in_control = cand_dist[i] <= cfg.control_radius_m
        c = int(cand_ids[i]) if in_control else -1
        if c == cur_holder:
            challenge_len = 0
        elif c == -1:
            # ball loose: keep possession attributed to current holder's team
            challenge_len = 0
        else:
            if c == challenger:
                challenge_len += 1
            else:
                challenger, challenge_len = c, 1
            if challenge_len >= cfg.hysteresis_frames or cur_holder == -1:
                cur_holder, cur_team = c, str(cand_team[i])
                challenge_len = 0
        holder[i] = cur_holder
        team[i] = cur_team
    return pd.DataFrame({"frame": frames, "holder_id": holder, "team": team,
                         "dist_m": cand_dist})


def possession_spells(possession: pd.DataFrame) -> pd.DataFrame:
    """Contiguous same-holder spells: holder_id, team, start_frame, end_frame."""
    p = possession.sort_values("frame")
    change = (p["holder_id"] != p["holder_id"].shift()).cumsum()
    spells = p.groupby(change).agg(
        holder_id=("holder_id", "first"),
        team=("team", "first"),
        start_frame=("frame", "first"),
        end_frame=("frame", "last"),
    ).reset_index(drop=True)
    return spells[spells.holder_id != -1].reset_index(drop=True)


def possession_summary(possession: pd.DataFrame, fps: float) -> dict:
    """Team possession shares and spell statistics."""
    contested = possession[possession["team"] != "none"]
    shares = contested["team"].value_counts(normalize=True).to_dict()
    spells = possession_spells(possession)
    spells["dur_s"] = (spells.end_frame - spells.start_frame + 1) / fps
    by_team = spells.groupby("team")["dur_s"]
    return {
        "share": {k: round(float(v), 4) for k, v in shares.items()},
        "n_spells": int(len(spells)),
        "avg_spell_s": {k: round(float(v), 2) for k, v in by_team.mean().to_dict().items()},
        "longest_spell_s": {k: round(float(v), 2) for k, v in by_team.max().to_dict().items()},
    }
