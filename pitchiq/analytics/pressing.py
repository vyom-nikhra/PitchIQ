"""Pressing intensity: pressure events, PPDA, press triggers & locations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitchiq.config import PressingConfig
from pitchiq.core.schema import MatchMeta
from pitchiq.analytics.common import attack_sign_series, to_attacking_x


def pressure_events(kin: pd.DataFrame, possession: pd.DataFrame,
                    team_of: dict[int, str], fps: float,
                    cfg: PressingConfig) -> pd.DataFrame:
    """A pressure: an opponent within ``press_radius_m`` of the ball holder AND
    closing (approach speed above threshold). Merged so one approach = one event."""
    poss = possession[possession.holder_id >= 0][["frame", "holder_id", "team"]]
    kin_idx = kin.set_index(["frame", "entity_id"])
    rows = []
    for _, p in poss.iterrows():
        f, holder = int(p.frame), int(p.holder_id)
        try:
            hx, hy = kin_idx.loc[(f, holder), ["x", "y"]]
        except KeyError:
            continue
        frame_players = kin[kin.frame == f]
        for _, q in frame_players.iterrows():
            eid = int(q.entity_id)
            if team_of.get(eid) in (None, "none", str(p.team)):
                continue
            d = float(np.hypot(q.x - hx, q.y - hy))
            if d > cfg.press_radius_m:
                continue
            # closing speed: velocity component toward the holder
            if d > 1e-6:
                closing = float((-(q.x - hx) * q.vx - (q.y - hy) * q.vy) / d)
            else:
                closing = 0.0
            if closing >= cfg.closing_speed_mps or d <= cfg.intense_press_dist_m:
                rows.append(dict(frame=f, presser_id=eid,
                                 presser_team=team_of.get(eid),
                                 target_id=holder, dist_m=round(d, 2),
                                 closing_mps=round(closing, 2), x=hx, y=hy))
    press = pd.DataFrame(rows)
    if press.empty:
        return press
    # merge contiguous same-pair frames into single events
    press = press.sort_values(["presser_id", "target_id", "frame"])
    gap = press.groupby(["presser_id", "target_id"])["frame"].diff().fillna(99)
    press["event_id"] = (gap > fps * 0.6).cumsum()
    merged = press.groupby("event_id").agg(
        frame=("frame", "first"), presser_id=("presser_id", "first"),
        presser_team=("presser_team", "first"), target_id=("target_id", "first"),
        dist_m=("dist_m", "min"), closing_mps=("closing_mps", "max"),
        x=("x", "mean"), y=("y", "mean"), duration_frames=("frame", "size"),
    ).reset_index(drop=True)
    return merged


def ppda(events: pd.DataFrame, press: pd.DataFrame, meta: MatchMeta,
         cfg: PressingConfig) -> dict:
    """PPDA per team: opponent passes in their build-up zone divided by our
    defensive actions (pressures + turnovers won) in that zone. Lower = more
    aggressive press. Zone = the opponent's defensive ``ppda_zone_frac`` of
    the pitch, measured in the opponent's attacking frame."""
    out = {}
    for team in ("home", "away"):
        opp = "away" if team == "home" else "home"
        opp_passes = events[(events.type == "pass") & (events.team == opp)]
        if len(opp_passes):
            signs = attack_sign_series(meta, opp_passes.frame.to_numpy(), opp)
            x_att = to_attacking_x(opp_passes.x.to_numpy(), signs, meta.pitch_length)
            n_passes = int((x_att <= cfg.ppda_zone_frac * meta.pitch_length).sum())
        else:
            n_passes = 0
        n_def = 0
        if len(press):
            our_press = press[press.presser_team == team]
            if len(our_press):
                signs = attack_sign_series(meta, our_press.frame.to_numpy(), opp)
                x_att = to_attacking_x(our_press.x.to_numpy(), signs, meta.pitch_length)
                n_def += int((x_att <= cfg.ppda_zone_frac * meta.pitch_length).sum())
        turnovers = events[(events.type == "turnover") & (events.team == team)]
        if len(turnovers):
            signs = attack_sign_series(meta, turnovers.frame.to_numpy(), opp)
            x_att = to_attacking_x(turnovers.x.to_numpy(), signs, meta.pitch_length)
            n_def += int((x_att <= cfg.ppda_zone_frac * meta.pitch_length).sum())
        out[team] = {
            "ppda": round(n_passes / n_def, 2) if n_def else float("inf"),
            "opp_buildup_passes": n_passes,
            "defensive_actions": n_def,
        }
    return out


def pressing_summary(press: pd.DataFrame, events: pd.DataFrame, meta: MatchMeta,
                     fps: float, n_frames: int, cfg: PressingConfig) -> dict:
    """Team pressing profile: volume, intensity, where, and win rate."""
    minutes = n_frames / fps / 60
    out = {}
    for team in ("home", "away"):
        tp = press[press.presser_team == team] if len(press) else pd.DataFrame()
        opp = "away" if team == "home" else "home"
        entry: dict = {
            "pressures": int(len(tp)),
            "pressures_per_min": round(len(tp) / minutes, 2) if minutes else 0,
        }
        if len(tp):
            signs = attack_sign_series(meta, tp.frame.to_numpy(), team)
            x_att = to_attacking_x(tp.x.to_numpy(), signs, meta.pitch_length)
            entry["press_height_mean_m"] = round(float(x_att.mean()), 1)
            entry["high_press_share"] = round(float((x_att > 2 * meta.pitch_length / 3).mean()), 3)
            # press→turnover within 3 s
            turnover_frames = events[(events.type == "turnover")
                                     & (events.team == team)].frame.to_numpy()
            wins = sum(bool(((turnover_frames >= f) & (turnover_frames <= f + 3 * fps)).any())
                       for f in tp.frame)
            entry["press_to_turnover_rate"] = round(wins / len(tp), 3)
        out[team] = entry
    return out
