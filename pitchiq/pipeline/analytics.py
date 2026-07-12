"""Layer-2 orchestration: tracking table → all analytics artifacts.

Pure dataframe work — never touches video, so it reruns in seconds on a
cached tracking table. Every product is persisted through the ArtifactStore:

    analytics/kinematics.parquet    per-frame smoothed positions/velocities
    analytics/possession.parquet    per-frame holder
    events.parquet                  derived passes/turnovers/carries
    analytics/heatmaps.npz          player + team occupancy grids
    analytics/pitch_control.npz     mean control surface (+ per-frame shares)
    analytics/phases.parquet        per-frame phase labels
    analytics/summary.json          everything scalar, grouped by module
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from pitchiq.analytics import (
    events as events_mod,
    formations as formations_mod,
    heatmaps as heatmaps_mod,
    kinematics as kin_mod,
    passes as passes_mod,
    phases as phases_mod,
    pitch_control as pc_mod,
    possession as poss_mod,
    pressing as press_mod,
    shape as shape_mod,
    xt as xt_mod,
)
from pitchiq.analytics.common import (
    ball_series,
    jersey_of_entities,
    minutes_played,
    team_of_entities,
)
from pitchiq.config import Config
from pitchiq.core.artifacts import ArtifactStore

log = logging.getLogger(__name__)


class AnalyticsPipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self, store: ArtifactStore,
            progress_cb: Callable[[float, str], None] | None = None) -> dict:
        def tick(p: float, msg: str) -> None:
            log.info("analytics %.0f%% — %s", 100 * p, msg)
            if progress_cb:
                progress_cb(p, msg)

        df = store.load_tracking()
        meta = store.load_meta()
        fps = meta.fps
        summary: dict = {"teams": meta.team_names}

        tick(0.05, "kinematics")
        kin = kin_mod.compute_kinematics(df, fps, self.cfg.kinematics)
        kin.to_parquet(store.analytics_path("kinematics.parquet"), index=False)
        profiles = kin_mod.player_movement_profiles(kin, fps, self.cfg.kinematics)
        team_of = team_of_entities(df)
        jersey_of = jersey_of_entities(df)
        summary["players"] = {
            str(eid): {"team": team_of.get(eid, "none"),
                       "jersey_no": jersey_of.get(eid), **prof}
            for eid, prof in profiles.items()
        }
        summary["team_distance_m"] = {
            t: round(sum(p["distance_m"] for e, p in profiles.items()
                         if team_of.get(e) == t), 0)
            for t in ("home", "away")
        }

        tick(0.2, "possession")
        possession = poss_mod.compute_possession(df, fps, self.cfg.possession)
        possession.to_parquet(store.analytics_path("possession.parquet"), index=False)
        summary["possession"] = poss_mod.possession_summary(possession, fps)

        tick(0.3, "events (passes/turnovers/carries)")
        events = events_mod.derive_events(
            df, possession, fps, self.cfg.passes,
            control_radius_m=self.cfg.possession.control_radius_m)
        store.save_events(events)
        counters = events_mod.detect_counter_attacks(events, ball_series(df), meta, fps)
        summary["events"] = {
            "n_passes": int((events.type == "pass").sum()),
            "n_completed_passes": int(((events.type == "pass")
                                       & (events.outcome == "complete")).sum()),
            "n_turnovers": int((events.type == "turnover").sum()),
            "n_carries": int((events.type == "carry").sum()),
            "counter_attacks": counters,
        }

        tick(0.4, "heatmaps")
        hm = heatmaps_mod.all_heatmaps(df, self.cfg.heatmaps, meta.pitch_length, meta.pitch_width)
        store.save_npz(store.analytics_path("heatmaps.npz"), **hm)
        summary["territory"] = heatmaps_mod.third_occupation(df, meta.pitch_length)

        tick(0.5, "formations")
        fw = formations_mod.formation_windows(df, possession, meta, self.cfg.formations)
        fw.to_parquet(store.analytics_path("formation_windows.parquet"), index=False)
        summary["formations"] = formations_mod.formation_summary(fw)

        tick(0.6, "pitch control")
        kin_pc = kin
        if self.cfg.pitch_control.impute_offscreen:
            kin_pc = pc_mod.impute_offscreen(
                kin, self.cfg.pitch_control.impute_horizon_s, meta.fps)
            summary["pitch_control_note"] = (
                "off-screen players imputed as decaying ghosts "
                f"(horizon {self.cfg.pitch_control.impute_horizon_s:.0f}s)")
        mean_grid, pc_frames = pc_mod.mean_control(kin_pc, team_of, self.cfg.pitch_control,
                                                   meta.pitch_length, meta.pitch_width)
        store.save_npz(store.analytics_path("pitch_control.npz"),
                       mean_home_control=mean_grid)
        pc_frames.to_parquet(store.analytics_path("pitch_control_frames.parquet"), index=False)
        summary["pitch_control"] = {
            "home_mean_control": round(float(mean_grid.mean()), 3),
            "home_final_third_control": round(float(np.array_split(mean_grid, 3, axis=1)[2].mean()), 3),
        }

        tick(0.7, "shape & field tilt")
        shape_ts = shape_mod.shape_timeseries(df, possession, meta)
        shape_ts.to_parquet(store.analytics_path("shape.parquet"), index=False)
        summary["shape"] = shape_mod.shape_summary(shape_ts)
        summary["field_tilt"] = shape_mod.field_tilt(df, possession, meta)

        tick(0.78, "pass networks & line-breaking passes")
        summary["pass_network"] = {
            t: passes_mod.build_pass_network(events, df, t) for t in ("home", "away")
        }
        lbp = passes_mod.line_breaking_passes(events, df, meta)
        lbp.to_parquet(store.analytics_path("line_breaking.parquet"), index=False)
        summary["line_breaking"] = {
            "total": int(len(lbp)),
            "by_team": {k: int(v) for k, v in
                        (lbp.team.value_counts().to_dict() if len(lbp) else {}).items()},
        }

        tick(0.86, "expected threat")
        grid = xt_mod.load_or_fit_grid(events, meta, self.cfg.xt)
        store.save_npz(store.analytics_path("xt_grid.npz"), xt=grid)
        xt_players = xt_mod.player_xt_contributions(events, grid, meta, self.cfg.xt)
        xt_players.to_parquet(store.analytics_path("xt_players.parquet"), index=False)
        summary["xt"] = {
            "top_players": xt_players.head(6).to_dict(orient="records"),
            "team_created": {k: round(float(v), 3) for k, v in
                             xt_players.groupby("team")["xt_created"].sum().to_dict().items()},
        }

        tick(0.92, "pressing / PPDA")
        press = press_mod.pressure_events(kin, possession, team_of, fps, self.cfg.pressing)
        if len(press):
            press.to_parquet(store.analytics_path("pressures.parquet"), index=False)
        summary["ppda"] = press_mod.ppda(events, press, meta, self.cfg.pressing)
        summary["pressing"] = press_mod.pressing_summary(
            press, events, meta, fps, meta.n_frames, self.cfg.pressing)

        tick(0.97, "phase segmentation")
        phases = phases_mod.segment_phases(df, possession, meta, self.cfg.phases)
        phases.to_parquet(store.analytics_path("phases.parquet"), index=False)
        summary["phases"] = phases_mod.phase_summary(phases, fps)

        store.save_json(store.analytics_path("summary.json"), summary)
        tick(1.0, "analytics complete")
        return summary
