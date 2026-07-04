"""Event derivation from tracking: passes, turnovers, carries.

Tracking-only event detection (no event feed): possession spells from the
hysteresis possession model are stitched into events —

* consecutive spells, same team, different holder → **pass** (complete)
* consecutive spells, different team → **turnover**; if the ball travelled
  ≥ ``min_pass_dist_m`` between spells it is also recorded as an
  **intercepted pass** by the losing team
* movement ≥ ``min_carry_dist_m`` within one spell → **carry**

Validated against the simulator's ground-truth pass log (precision/recall in
``scripts/validate_synthetic.py``); on real footage accuracy is bounded by
ball-tracking quality — the documented weakest perception link.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitchiq.config import PassesConfig
from pitchiq.analytics.common import ball_series
from pitchiq.analytics.possession import possession_spells

EVENT_COLUMNS = [
    "frame", "timestamp", "type", "team", "from_id", "to_id",
    "x", "y", "end_x", "end_y", "outcome",
]


def derive_events(df: pd.DataFrame, possession: pd.DataFrame, fps: float,
                  cfg: PassesConfig, control_radius_m: float = 2.0) -> pd.DataFrame:
    """Tracking-derived event table (schema-compatible with the simulator's).

    Spell boundaries lag the physical pass by the possession hysteresis, so a
    pass's origin is anchored at the *last frame the passer actually held the
    ball* (nearest-player distance ≤ control radius) and its end at the first
    truly-held frame of the next spell — otherwise start≈end and every pass
    fails the minimum-distance gate.
    """
    spells = possession_spells(possession)
    ball = ball_series(df)
    persons = df[df["class"].isin(["player", "goalkeeper"])].dropna(
        subset=["x_pitch", "y_pitch"])
    pos_lookup = persons.set_index(["frame", "entity_id"])[["x_pitch", "y_pitch"]].sort_index()
    events: list[dict] = []

    def ball_at(frame: int) -> tuple[float, float]:
        if frame in ball.index:
            row = ball.loc[frame]
            return float(row.x_pitch), float(row.y_pitch)
        idx = ball.index[np.argmin(np.abs(ball.index.to_numpy() - frame))]
        row = ball.loc[idx]
        return float(row.x_pitch), float(row.y_pitch)

    def dist_to_holder(frame: int, holder: int) -> float:
        """Ball distance to a SPECIFIC player (not nearest-player: during
        flight the ball passes close to bystanders, which must not count
        as possession)."""
        try:
            px, py = pos_lookup.loc[(frame, holder)]
        except KeyError:
            return float("inf")
        bx, by = ball_at(frame)
        return float(np.hypot(bx - px, by - py))

    def last_held_frame(spell) -> int:
        for f in range(int(spell.end_frame), int(spell.start_frame) - 1, -1):
            if dist_to_holder(f, int(spell.holder_id)) <= control_radius_m:
                return f
        return int(spell.end_frame)

    def first_held_frame(spell) -> int:
        for f in range(int(spell.start_frame), int(spell.end_frame) + 1):
            if dist_to_holder(f, int(spell.holder_id)) <= control_radius_m:
                return f
        return int(spell.start_frame)

    for i in range(len(spells) - 1):
        cur = spells.iloc[i]
        nxt = spells.iloc[i + 1]
        release_f = last_held_frame(cur)
        receive_f = first_held_frame(nxt)
        gap_s = (receive_f - release_f) / fps
        x0, y0 = ball_at(release_f)
        x1, y1 = ball_at(receive_f)
        dist = float(np.hypot(x1 - x0, y1 - y0))
        ts = float(release_f / fps)

        # carry within the current spell
        sx, sy = ball_at(first_held_frame(cur))
        carry_d = float(np.hypot(x0 - sx, y0 - sy))
        if carry_d >= cfg.min_carry_dist_m:
            carry_f = first_held_frame(cur)
            events.append(dict(
                frame=carry_f, timestamp=float(carry_f / fps),
                type="carry", team=str(cur.team), from_id=int(cur.holder_id),
                to_id=int(cur.holder_id), x=sx, y=sy, end_x=x0, end_y=y0,
                outcome="complete"))

        if gap_s > cfg.max_pass_time_s:
            continue  # dead ball / restart — not a pass
        if nxt.team == cur.team and nxt.holder_id != cur.holder_id:
            if dist >= cfg.min_pass_dist_m:
                events.append(dict(
                    frame=release_f, timestamp=ts, type="pass",
                    team=str(cur.team), from_id=int(cur.holder_id),
                    to_id=int(nxt.holder_id), x=x0, y=y0, end_x=x1, end_y=y1,
                    outcome="complete"))
        elif nxt.team != cur.team:
            events.append(dict(
                frame=receive_f, timestamp=float(receive_f / fps),
                type="turnover", team=str(nxt.team), from_id=int(cur.holder_id),
                to_id=int(nxt.holder_id), x=x1, y=y1, end_x=np.nan, end_y=np.nan,
                outcome="interception" if dist >= cfg.min_pass_dist_m else "tackle"))
            if dist >= cfg.min_pass_dist_m:
                events.append(dict(
                    frame=release_f, timestamp=ts, type="pass",
                    team=str(cur.team), from_id=int(cur.holder_id), to_id=-1,
                    x=x0, y=y0, end_x=x1, end_y=y1, outcome="intercepted"))
    ev = pd.DataFrame(events, columns=EVENT_COLUMNS)
    return ev.sort_values("frame").reset_index(drop=True)


def detect_counter_attacks(events: pd.DataFrame, ball: pd.DataFrame, meta,
                           fps: float, window_s: float = 8.0,
                           min_gain_m: float = 25.0) -> list[dict]:
    """Fast defence→attack transitions: within ``window_s`` of a turnover the
    winning team advances the ball ≥ ``min_gain_m`` toward the opponent goal."""
    from pitchiq.analytics.common import attack_sign_series, to_attacking_x

    out = []
    turnovers = events[events.type == "turnover"]
    for _, t in turnovers.iterrows():
        f0 = int(t.frame)
        f1 = f0 + int(window_s * fps)
        seg = ball.loc[(ball.index >= f0) & (ball.index <= f1)]
        if len(seg) < 5:
            continue
        signs = attack_sign_series(meta, seg.index.to_numpy(), str(t.team))
        x_att = to_attacking_x(seg["x_pitch"].to_numpy(), signs, meta.pitch_length)
        gain = float(np.nanmax(x_att) - x_att[0]) if np.isfinite(x_att).any() else 0.0
        if gain >= min_gain_m:
            dur = (seg.index[int(np.nanargmax(x_att))] - f0) / fps
            out.append(dict(frame=f0, team=str(t.team), gain_m=round(gain, 1),
                            duration_s=round(float(dur), 2),
                            speed_m_per_s=round(gain / max(dur, 0.4), 2)))
    return out
