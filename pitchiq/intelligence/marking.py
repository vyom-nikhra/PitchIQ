"""Marking analysis: man-to-man vs zonal, and who-marks-whom.

Per defensive-phase sample, the optimal defender→attacker assignment is
solved with the Hungarian algorithm on pairwise distance (gated by
``max_pair_dist_m``). From the assignment timeline:

* **assignment stability** — how consistently a defender is assigned to the
  same attacker over a sliding window (man-marking is stable, zonal rotates)
* **trajectory correlation** — Pearson correlation of defender vs assigned
  attacker velocities over stable spells (a man-marker mirrors their man's
  movement; a zonal defender doesn't)
* **spatial stability** — a zonal defender holds a region (low positional
  spread through the defensive phase) even while assignments churn

Each defender gets a man-score in [0,1] (spectrum, not binary — most teams
are hybrid); the team score aggregates outfield defenders. Marking pairs are
reported for defenders above the stability threshold, split by open play vs
set pieces (set-piece marking is often man even for zonal teams).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from pitchiq.config import MarkingConfig


def assignment_timeline(
    kin: pd.DataFrame,
    phases: pd.DataFrame,
    team_of: dict[int, str],
    defending_team: str,
    cfg: MarkingConfig,
    fps: float,
    phase_filter: str = "open_play",
    exclude_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Hungarian defender→attacker assignments over defensive-phase samples.

    Columns: frame, defender_id, attacker_id, dist_m.
    ``phase_filter``: 'open_play' | 'set_piece' | 'all'.
    ``exclude_ids``: entities never assigned (goalkeepers, by default upstream).
    """
    attacking = "home" if defending_team == "away" else "away"
    ph = phases[(phases.poss_team == attacking)]
    if phase_filter == "open_play":
        ph = ph[ph.phase != "set_piece"]
    elif phase_filter == "set_piece":
        ph = ph[ph.phase == "set_piece"]
    frames = ph.frame.to_numpy()
    step = max(1, int(cfg.step_s * fps))
    frames = frames[::step] if len(frames) else frames

    kin_by_frame = dict(tuple(kin.groupby("frame")))
    excl = exclude_ids or set()
    rows = []
    for f in frames:
        g = kin_by_frame.get(int(f))
        if g is None:
            continue
        g = g[~g.entity_id.isin(excl)]
        teams = g["entity_id"].map(team_of)
        defs = g[teams == defending_team]
        atts = g[teams == attacking]
        if len(defs) < 5 or len(atts) < 5:
            continue
        D = defs[["x", "y"]].to_numpy()
        A = atts[["x", "y"]].to_numpy()
        cost = np.linalg.norm(D[:, None] - A[None], axis=2)
        gated = np.where(cost > cfg.max_pair_dist_m, cost + 1e5, cost)
        ri, ci = linear_sum_assignment(gated)
        for r, c in zip(ri, ci):
            if cost[r, c] > cfg.max_pair_dist_m:
                continue
            rows.append(dict(frame=int(f),
                             defender_id=int(defs.iloc[r].entity_id),
                             attacker_id=int(atts.iloc[c].entity_id),
                             dist_m=round(float(cost[r, c]), 2)))
    return pd.DataFrame(rows)


def _stability(assign: pd.DataFrame, window_samples: int) -> dict[int, float]:
    """Per defender: mean share of the modal assignment over sliding windows."""
    out = {}
    for did, g in assign.groupby("defender_id"):
        seq = g.sort_values("frame")["attacker_id"].to_numpy()
        if len(seq) < max(4, window_samples // 2):
            continue
        w = min(window_samples, len(seq))
        scores = []
        for i in range(0, len(seq) - w + 1, max(1, w // 2)):
            win = seq[i: i + w]
            _, counts = np.unique(win, return_counts=True)
            scores.append(counts.max() / len(win))
        out[int(did)] = float(np.mean(scores))
    return out


def _position_coupling(assign: pd.DataFrame, kin: pd.DataFrame,
                       team_of: dict[int, str], phases: pd.DataFrame,
                       defending_team: str, min_len: int = 30) -> dict[int, float]:
    """RESIDUAL-POSITION coupling between defender and modal attacker.

    Both a man-marker and a zonal block shift with the ball, so raw motion
    similarity is high for everyone. Subtract each side's per-frame team
    centroid: a zonal defender's residual position is their (fixed) slot in
    the block regardless of where their nearest attacker roams, while a
    man-marker's residual position *tracks* the attacker's residual. Position
    series are far less noisy than instantaneous velocities, so this is
    computed densely over all defensive-phase frames, not just Hungarian
    samples.
    """
    attacking = "home" if defending_team == "away" else "away"
    def_frames = set(phases[phases.poss_team == attacking].frame.tolist())
    sub = kin[kin.frame.isin(def_frames)].copy()
    sub["team"] = sub["entity_id"].map(team_of)
    sub = sub[sub.team.isin(["home", "away"])]
    cent = sub.groupby(["frame", "team"])[["x", "y"]].transform("mean")
    sub["rx"] = sub["x"] - cent["x"]
    sub["ry"] = sub["y"] - cent["y"]
    kin_idx = sub.set_index(["frame", "entity_id"])[["rx", "ry"]].sort_index()

    out = {}
    for did, g in assign.groupby("defender_id"):
        modal = g["attacker_id"].mode()
        if not len(modal):
            continue
        aid = int(modal.iat[0])
        try:
            d_ser = kin_idx.xs(int(did), level="entity_id")
            a_ser = kin_idx.xs(aid, level="entity_id")
        except KeyError:
            continue
        joined = d_ser.join(a_ser, how="inner", lsuffix="_d", rsuffix="_a")
        if len(joined) < min_len:
            continue
        dv = np.concatenate([joined["rx_d"], joined["ry_d"]])
        av = np.concatenate([joined["rx_a"], joined["ry_a"]])
        # centre each series (residual positions have arbitrary offsets)
        dv = dv - dv.mean()
        av = av - av.mean()
        if dv.std() < 1e-6 or av.std() < 1e-6:
            out[int(did)] = 0.0
            continue
        out[int(did)] = float(np.corrcoef(dv, av)[0, 1])
    return out


def _spatial_spread(kin: pd.DataFrame, phases: pd.DataFrame, team_of: dict[int, str],
                    defending_team: str) -> dict[int, float]:
    attacking = "home" if defending_team == "away" else "away"
    def_frames = set(phases[phases.poss_team == attacking].frame.tolist())
    sub = kin[kin.frame.isin(def_frames)]
    out = {}
    for eid, g in sub.groupby("entity_id"):
        if team_of.get(int(eid)) != defending_team or len(g) < 10:
            continue
        out[int(eid)] = float(np.hypot(g.x.std(), g.y.std()))
    return out


def analyse_marking(
    kin: pd.DataFrame,
    phases: pd.DataFrame,
    team_of: dict[int, str],
    cfg: MarkingConfig,
    fps: float,
    class_of: dict[int, str] | None = None,
) -> dict:
    """Full marking artifact for both teams (+ per-phase split + pairs).

    Goalkeepers are excluded from assignments on both sides — a GK "marking"
    the nearest striker is an artifact, not a scheme signal.
    """
    gk_ids = {eid for eid, c in (class_of or {}).items() if c == "goalkeeper"}
    result: dict = {}
    for team in ("home", "away"):
        entry: dict = {}
        for phase_filter in ("open_play", "set_piece"):
            assign = assignment_timeline(kin, phases, team_of, team, cfg, fps,
                                         phase_filter, exclude_ids=gk_ids)
            if len(assign) < max(6, cfg.min_defensive_frames // max(1, int(cfg.step_s * fps))):
                entry[phase_filter] = {"note": "not enough defensive samples"}
                continue
            window_samples = max(3, int(cfg.window_s / cfg.step_s))
            stability = _stability(assign, window_samples)
            corr = _position_coupling(assign, kin, team_of, phases, team)
            spread = _spatial_spread(kin, phases, team_of, team)

            per_defender = {}
            for did, stab in stability.items():
                c = corr.get(did, 0.0)
                # residual-position coupling is the discriminator; stability gates it
                man_score = float(np.clip(stab * (0.25 + 0.75 * np.clip(c, 0, 1)), 0, 1))
                per_defender[str(did)] = {
                    "assignment_stability": round(stab, 3),
                    "position_coupling": round(c, 3),
                    "spatial_spread_m": round(spread.get(did, np.nan), 2)
                    if did in spread else None,
                    "man_score": round(man_score, 3),
                }
            man_scores = [d["man_score"] for d in per_defender.values()]
            team_score = float(np.mean(man_scores)) if man_scores else 0.0
            pairs = []
            for did, g in assign.groupby("defender_id"):
                if stability.get(int(did), 0.0) < cfg.stability_threshold:
                    continue
                modal = g["attacker_id"].mode()
                if not len(modal):
                    continue
                aid = int(modal.iat[0])
                share = float((g["attacker_id"] == aid).mean())
                pairs.append({"defender_id": int(did), "attacker_id": aid,
                              "share": round(share, 3),
                              "avg_dist_m": round(float(
                                  g.loc[g.attacker_id == aid, "dist_m"].mean()), 2)})
            entry[phase_filter] = {
                "team_man_score": round(team_score, 3),
                # a spectrum, not a binary: aggressive pressing teams score
                # mid-range because chasing the carrier IS man-oriented
                "scheme": ("man-marking" if team_score >= 0.72
                           else "zonal" if team_score <= 0.50 else "hybrid"),
                "per_defender": per_defender,
                "pairs": sorted(pairs, key=lambda p: -p["share"]),
                "n_samples": int(assign.frame.nunique()),
            }
        result[team] = entry
    return result
