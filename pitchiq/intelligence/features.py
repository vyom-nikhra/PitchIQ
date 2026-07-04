"""Player-style features: movement & involvement, not box-score stats.

For every outfield player we build named feature groups (kept separate so
similarity search can attribute *why* two players are alike):

* ``spatial``      — downsampled positional heatmap + territory statistics,
  in the player's attacking frame (so left-backs of both teams align)
* ``movement``     — speed/accel distributions, sprint & direction-change rates
* ``involvement``  — touches, possession share, distance-to-ball, pressing rate,
  time share per phase
* ``interaction``  — teammate spacing + pass-network centralities
* ``phase``        — position/speed conditioned on phase (attack vs defence
  behaviour differs; that difference IS the signal)

All rates are per-minute so partial appearances compare fairly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pitchiq.config import EmbeddingsConfig
from pitchiq.core.schema import MatchMeta
from pitchiq.analytics.common import attack_sign_series, to_attacking_xy
from pitchiq.analytics.heatmaps import position_heatmap
from pitchiq.config import HeatmapConfig

PHASE_KEYS = ["build_up", "progression", "final_third_attack", "transition_attack",
              "set_piece", "defending"]


@dataclass
class PlayerFeatures:
    entity_id: int
    team: str
    minutes: float
    groups: dict[str, dict[str, float]] = field(default_factory=dict)
    heatmap: np.ndarray | None = None          # (ny, nx) attacking frame
    heatmap_attack: np.ndarray | None = None   # in-possession only
    heatmap_defend: np.ndarray | None = None   # out-of-possession only

    def flat(self) -> tuple[np.ndarray, list[str]]:
        """Scalar features flattened (heatmap appended separately by embeddings)."""
        names, vals = [], []
        for gname in sorted(self.groups):
            for fname in sorted(self.groups[gname]):
                names.append(f"{gname}.{fname}")
                vals.append(self.groups[gname][fname])
        return np.array(vals, dtype=np.float64), names


def extract_player_features(
    df: pd.DataFrame,
    kin: pd.DataFrame,
    possession: pd.DataFrame,
    events: pd.DataFrame,
    phases: pd.DataFrame,
    passnet: dict,
    meta: MatchMeta,
    cfg: EmbeddingsConfig,
    movement_profiles: dict[int, dict] | None = None,
    press_events: pd.DataFrame | None = None,
    min_minutes: float = 1.0,
) -> dict[int, PlayerFeatures]:
    """Build per-player feature groups for all qualifying outfield players."""
    fps = meta.fps
    hm_cfg = HeatmapConfig(nx=cfg.heatmap_nx, ny=cfg.heatmap_ny, sigma_cells=1.0)
    outfield = df[(df["class"] == "player") & df.team.isin(["home", "away"])].dropna(
        subset=["x_pitch", "y_pitch"])

    poss_by_frame = possession.set_index("frame")
    phase_by_frame = phases.set_index("frame")["phase"] if len(phases) else pd.Series(dtype=object)
    poss_team_by_frame = phases.set_index("frame")["poss_team"] if len(phases) else pd.Series(dtype=object)

    # ball distance per row (vectorised)
    ball = df[df.entity_id == -1].set_index("frame")[["x_pitch", "y_pitch"]]
    ball = ball.rename(columns={"x_pitch": "bx", "y_pitch": "by"})

    # teammate spacing per frame per team
    centroid = outfield.groupby(["frame", "team"])[["x_pitch", "y_pitch"]].mean()

    # pass-network lookups
    net_nodes: dict[int, dict] = {}
    for t in ("home", "away"):
        for node in passnet.get(t, {}).get("nodes", []):
            net_nodes[int(node["id"])] = node

    out: dict[int, PlayerFeatures] = {}
    for eid, g in outfield.groupby("entity_id"):
        eid = int(eid)
        team = str(g["team"].mode().iat[0])
        minutes = g["frame"].nunique() / fps / 60.0
        if minutes < min_minutes:
            continue
        frames = g["frame"].to_numpy()
        signs = attack_sign_series(meta, frames, team)
        xy_att = to_attacking_xy(g[["x_pitch", "y_pitch"]].to_numpy(), signs,
                                 meta.pitch_length, meta.pitch_width)

        pf = PlayerFeatures(entity_id=eid, team=team, minutes=round(minutes, 2))

        # ---------------- spatial ----------------
        hull = _hull_area(xy_att)
        pf.groups["spatial"] = {
            "mean_x": float(xy_att[:, 0].mean()),
            "mean_y": float(xy_att[:, 1].mean()),
            "std_x": float(xy_att[:, 0].std()),
            "std_y": float(xy_att[:, 1].std()),
            "wideness": float(np.abs(xy_att[:, 1] - meta.pitch_width / 2).mean()),
            "hull_area": hull,
            "x_range": float(np.percentile(xy_att[:, 0], 95) - np.percentile(xy_att[:, 0], 5)),
            "y_range": float(np.percentile(xy_att[:, 1], 95) - np.percentile(xy_att[:, 1], 5)),
        }
        pf.heatmap = position_heatmap(xy_att[:, 0], xy_att[:, 1], hm_cfg,
                                      meta.pitch_length, meta.pitch_width)

        # attack/defence-split heatmaps (phase-conditioned imagery for 6.1b)
        if len(poss_team_by_frame):
            pt = poss_team_by_frame.reindex(frames).to_numpy(dtype=object)
            in_pos = pt == team
            out_pos = (pt != team) & (pt != "none")
            if in_pos.sum() > 10:
                pf.heatmap_attack = position_heatmap(
                    xy_att[in_pos, 0], xy_att[in_pos, 1], hm_cfg,
                    meta.pitch_length, meta.pitch_width)
            if out_pos.sum() > 10:
                pf.heatmap_defend = position_heatmap(
                    xy_att[out_pos, 0], xy_att[out_pos, 1], hm_cfg,
                    meta.pitch_length, meta.pitch_width)

        # ---------------- movement ----------------
        prof = (movement_profiles or {}).get(eid, {})
        kin_g = kin[kin.entity_id == eid]
        speeds = kin_g["speed"].dropna().to_numpy()
        pf.groups["movement"] = {
            "dist_per_min": prof.get("distance_per_min_m", 0.0),
            "top_speed": prof.get("top_speed_mps", 0.0),
            "speed_p50": float(np.percentile(speeds, 50)) if len(speeds) else 0.0,
            "speed_p90": float(np.percentile(speeds, 90)) if len(speeds) else 0.0,
            "sprints_per_min": prof.get("sprints_per_min", 0.0),
            "hi_dist_per_min": prof.get("hi_distance_m", 0.0) / max(minutes, 0.1),
            "accel_p90": prof.get("accel_p90", 0.0),
            "dir_changes_per_min": prof.get("direction_changes_per_min", 0.0),
        }

        # ---------------- involvement ----------------
        held = poss_by_frame[poss_by_frame.holder_id == eid]
        touches = _count_spells(held.index.to_numpy())
        gb = g.merge(ball, left_on="frame", right_index=True, how="left")
        dist_ball = np.hypot(gb.x_pitch - gb.bx, gb.y_pitch - gb.by).dropna()
        n_press = 0
        if press_events is not None and len(press_events):
            n_press = int((press_events.presser_id == eid).sum())
        phase_share = {}
        if len(phase_by_frame):
            ph = phase_by_frame.reindex(frames).dropna()
            pt = poss_team_by_frame.reindex(ph.index)
            for key in PHASE_KEYS[:-1]:
                phase_share[f"time_{key}"] = float(((ph == key) & (pt == team)).mean())
            phase_share["time_defending"] = float(((pt != team) & (pt != "none")).mean())
        pf.groups["involvement"] = {
            "touches_per_min": touches / minutes,
            "poss_time_share": float(len(held) / max(len(frames), 1)),
            "mean_dist_ball": float(dist_ball.mean()) if len(dist_ball) else 40.0,
            "near_ball_share": float((dist_ball < 15).mean()) if len(dist_ball) else 0.0,
            "press_per_min": n_press / minutes,
            **phase_share,
        }

        # ---------------- interaction ----------------
        cen = centroid.reset_index()
        cen_t = cen[cen.team == team].set_index("frame")[["x_pitch", "y_pitch"]]
        cc = g.merge(cen_t, left_on="frame", right_index=True, how="left",
                     suffixes=("", "_c")).dropna(subset=["x_pitch_c"])
        d_centroid = np.hypot(cc.x_pitch - cc.x_pitch_c, cc.y_pitch - cc.y_pitch_c)
        node = net_nodes.get(eid, {})
        pf.groups["interaction"] = {
            "dist_to_centroid": float(d_centroid.mean()) if len(d_centroid) else 15.0,
            "net_volume": float(node.get("volume", 0.0)) / max(minutes, 0.1),
            "net_betweenness": float(node.get("betweenness", 0.0)),
            "net_eigenvector": float(node.get("eigenvector", 0.0)),
        }

        # ---------------- phase-conditioned position ----------------
        phase_feats = {}
        if len(poss_team_by_frame):
            pt = poss_team_by_frame.reindex(frames).to_numpy(dtype=object)
            in_pos = pt == team
            out_pos = (pt != team) & (pt != "none")
            for label, mask in (("att", in_pos), ("def", out_pos)):
                if mask.sum() > 10:
                    phase_feats[f"{label}_mean_x"] = float(xy_att[mask, 0].mean())
                    phase_feats[f"{label}_mean_y"] = float(xy_att[mask, 1].mean())
                else:
                    phase_feats[f"{label}_mean_x"] = float(xy_att[:, 0].mean())
                    phase_feats[f"{label}_mean_y"] = float(xy_att[:, 1].mean())
            phase_feats["push_up_delta"] = phase_feats["att_mean_x"] - phase_feats["def_mean_x"]
            phase_feats["tuck_in_delta"] = abs(phase_feats["att_mean_y"] - meta.pitch_width / 2) \
                - abs(phase_feats["def_mean_y"] - meta.pitch_width / 2)
        pf.groups["phase"] = phase_feats

        out[eid] = pf
    return out


def _count_spells(frames: np.ndarray) -> int:
    """Number of separate holding spells in a set of held frames."""
    if len(frames) == 0:
        return 0
    frames = np.sort(frames)
    return int(1 + (np.diff(frames) > 3).sum())


def _hull_area(points: np.ndarray) -> float:
    from scipy.spatial import ConvexHull

    if len(points) < 4:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except Exception:
        return 0.0


def features_to_frame(features: dict[int, PlayerFeatures]) -> pd.DataFrame:
    """Flat table (one row per player) for persistence & inspection."""
    rows = []
    for eid, pf in features.items():
        vec, names = pf.flat()
        row = {"entity_id": eid, "team": pf.team, "minutes": pf.minutes}
        row.update(dict(zip(names, vec)))
        rows.append(row)
    return pd.DataFrame(rows)
