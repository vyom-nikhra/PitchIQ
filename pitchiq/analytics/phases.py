"""Phase segmentation: classify every frame into a game phase.

Rule-based classifier over possession + ball location + recency of turnover +
ball speed. Phases (from the possessing team's perspective):

    set_piece | build_up | progression | final_third_attack |
    transition_attack | contested

plus the defending team's posture (high_press / mid_block / low_block) from
its centroid depth. The intelligence layer conditions player-style features
on these phases, and marking analysis runs on the defensive ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitchiq.config import PhasesConfig
from pitchiq.core.schema import MatchMeta
from pitchiq.analytics.common import attack_sign_series, ball_series, to_attacking_x


def segment_phases(df: pd.DataFrame, possession: pd.DataFrame, meta: MatchMeta,
                   cfg: PhasesConfig) -> pd.DataFrame:
    """Per-frame phase table: frame, poss_team, phase, def_posture."""
    fps = meta.fps
    ball = ball_series(df)
    full_index = ball.index.to_numpy()
    bx = ball["x_pitch"].to_numpy()
    by = ball["y_pitch"].to_numpy()
    bspeed = np.hypot(np.gradient(np.nan_to_num(bx)), np.gradient(np.nan_to_num(by))) * fps

    poss = possession.set_index("frame").reindex(full_index)
    poss_team = poss["team"].fillna("none").to_numpy(dtype=object)

    # possession-change recency
    change_frames = full_index[np.r_[True, poss_team[1:] != poss_team[:-1]]]
    last_change = np.searchsorted(change_frames, full_index, side="right") - 1
    since_change_s = (full_index - change_frames[np.clip(last_change, 0, None)]) / fps

    # dead-ball detection: sustained near-zero ball speed
    dead_run = np.zeros(len(full_index))
    run = 0
    for i, s in enumerate(bspeed):
        run = run + 1 if s < cfg.dead_ball_speed_mps else 0
        dead_run[i] = run
    is_dead = dead_run >= cfg.dead_ball_secs * fps

    # defending-team centroid depth per frame (sampled, then ffilled)
    outfield = df[(df["class"] == "player") & df.team.isin(["home", "away"])]
    centroids = outfield.groupby(["frame", "team"])["x_pitch"].mean().unstack()

    phases = np.empty(len(full_index), dtype=object)
    postures = np.empty(len(full_index), dtype=object)
    for i, f in enumerate(full_index):
        team = str(poss_team[i])
        if team == "none":
            phases[i] = "contested"
            postures[i] = "none"
            continue
        sign = meta.attack_sign(team, int(f))
        x_att = bx[i] if sign > 0 else meta.pitch_length - bx[i]
        if is_dead[i]:
            phases[i] = "set_piece"
        elif since_change_s[i] <= cfg.transition_window_s and x_att > cfg.third_boundaries_m[0]:
            phases[i] = "transition_attack"
        elif x_att < cfg.third_boundaries_m[0]:
            phases[i] = "build_up"
        elif x_att < cfg.third_boundaries_m[1]:
            phases[i] = "progression"
        else:
            phases[i] = "final_third_attack"

        defender = "away" if team == "home" else "home"
        try:
            dx = centroids.loc[f, defender]
        except KeyError:
            dx = np.nan
        if np.isfinite(dx):
            dsign = meta.attack_sign(defender, int(f))
            d_att = dx if dsign > 0 else meta.pitch_length - dx  # defender's own frame
            postures[i] = ("high_press" if d_att > cfg.high_block_x_m * 0.68
                           else "low_block" if d_att < cfg.low_block_x_m * 0.85
                           else "mid_block")
        else:
            postures[i] = "mid_block"

    return pd.DataFrame({
        "frame": full_index,
        "poss_team": poss_team,
        "phase": phases,
        "def_posture": postures,
    })


def phase_summary(phases: pd.DataFrame, fps: float) -> dict:
    """Share of time per phase, per possessing team + posture shares."""
    out: dict = {"share_overall": {}}
    counts = phases["phase"].value_counts(normalize=True)
    out["share_overall"] = {k: round(float(v), 3) for k, v in counts.items()}
    for team in ("home", "away"):
        g = phases[phases.poss_team == team]
        if not len(g):
            continue
        out[f"{team}_in_possession"] = {
            k: round(float(v), 3)
            for k, v in g["phase"].value_counts(normalize=True).items()
        }
        opp = phases[(phases.poss_team != team) & (phases.poss_team != "none")]
        out[f"{team}_defending_posture"] = {
            k: round(float(v), 3)
            for k, v in opp["def_posture"].value_counts(normalize=True).items()
        }
    return out
