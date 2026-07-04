"""Layer-3 orchestration: analytics artifacts → intelligence artifacts.

    intelligence/features.parquet   flat per-player feature table
    intelligence/embeddings.npz     vectors + ids (+ scaled scalars)
    intelligence/roles.json         discovered roles, cluster traits, mismatches
    intelligence/similarity.json    top-k neighbours per player w/ attributions
    intelligence/marking.json       man/zonal spectrum, per-defender, pairs
    intelligence/marking_timeline.parquet  per-sample assignments (for the viz)
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from pitchiq.analytics.common import team_of_entities
from pitchiq.config import Config
from pitchiq.core.artifacts import ArtifactStore
from pitchiq.intelligence.embeddings import compute_embeddings
from pitchiq.intelligence.features import extract_player_features, features_to_frame
from pitchiq.intelligence.marking import analyse_marking, assignment_timeline
from pitchiq.intelligence.roles import discover_roles, nominal_vs_actual
from pitchiq.intelligence.similarity import SimilarityIndex

log = logging.getLogger(__name__)


class IntelligencePipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self, store: ArtifactStore,
            progress_cb: Callable[[float, str], None] | None = None) -> dict:
        def tick(p: float, msg: str) -> None:
            log.info("intelligence %.0f%% — %s", 100 * p, msg)
            if progress_cb:
                progress_cb(p, msg)

        df = store.load_tracking()
        meta = store.load_meta()
        summary = store.load_json(store.analytics_path("summary.json"))
        kin = pd.read_parquet(store.analytics_path("kinematics.parquet"))
        possession = pd.read_parquet(store.analytics_path("possession.parquet"))
        events = store.load_events()
        phases = pd.read_parquet(store.analytics_path("phases.parquet"))
        press_path = store.analytics_path("pressures.parquet")
        press = pd.read_parquet(press_path) if press_path.exists() else None
        fw_path = store.analytics_path("formation_windows.parquet")
        formation_windows = pd.read_parquet(fw_path) if fw_path.exists() else pd.DataFrame()

        movement_profiles = {int(k): v for k, v in summary.get("players", {}).items()}

        tick(0.1, "player-style features")
        features = extract_player_features(
            df, kin, possession, events, phases, summary.get("pass_network", {}),
            meta, self.cfg.embeddings, movement_profiles, press,
            min_minutes=self.cfg.roles.min_minutes,
        )
        features_to_frame(features).to_parquet(
            store.intelligence_path("features.parquet"), index=False)

        tick(0.35, "style embeddings")
        emb = compute_embeddings(features, self.cfg.embeddings)
        store.save_npz(store.intelligence_path("embeddings.npz"),
                       vectors=emb.vectors, ids=np.array(emb.ids),
                       scalars=emb.scalar_matrix)

        if len(emb.ids) == 0:
            log.warning("no players met the minutes threshold — skipping "
                        "roles/similarity (clip too short)")
            store.save_json(store.intelligence_path("roles.json"),
                            {"players": {}, "clusters": [],
                             "note": "clip too short for style profiling"})
            store.save_json(store.intelligence_path("similarity.json"),
                            {"backend": "none", "neighbors": {}})
            roles = {"players": {}, "nominal_vs_actual": []}
        else:
            tick(0.5, "role discovery")
            roles = discover_roles(emb, features, self.cfg.roles)
            roles["nominal_vs_actual"] = nominal_vs_actual(roles, formation_windows)
            store.save_json(store.intelligence_path("roles.json"), roles)

            tick(0.65, "similar-player search")
            sim = SimilarityIndex(emb, self.cfg.similarity)
            similarity = {"backend": sim.backend, "method": emb.method,
                          "neighbors": sim.all_neighbors()}
            store.save_json(store.intelligence_path("similarity.json"), similarity)

        tick(0.8, "marking analysis")
        team_of = team_of_entities(df)
        from pitchiq.analytics.common import class_of_entities

        class_of = class_of_entities(df)
        gk_ids = {eid for eid, c in class_of.items() if c == "goalkeeper"}
        marking = analyse_marking(kin, phases, team_of, self.cfg.marking, meta.fps,
                                  class_of=class_of)
        store.save_json(store.intelligence_path("marking.json"), marking)
        timelines = []
        for team in ("home", "away"):
            tl = assignment_timeline(kin, phases, team_of, team,
                                     self.cfg.marking, meta.fps, "open_play",
                                     exclude_ids=gk_ids)
            if len(tl):
                tl["defending_team"] = team
                timelines.append(tl)
        if timelines:
            pd.concat(timelines, ignore_index=True).to_parquet(
                store.intelligence_path("marking_timeline.parquet"), index=False)

        out = {
            "n_players_embedded": len(emb.ids),
            "embedding_method": emb.method,
            "roles": {str(k): v["role"] for k, v in roles.get("players", {}).items()},
            "role_mismatches": roles["nominal_vs_actual"],
            "marking": {t: {pf: (marking[t][pf].get("scheme"),
                                 marking[t][pf].get("team_man_score"))
                            for pf in marking[t]} for t in marking},
        }
        store.save_json(store.intelligence_path("summary.json"), out)
        tick(1.0, "intelligence complete")
        return out
