"""Unsupervised role discovery + human-readable naming.

Cluster the style embeddings (K-Means with silhouette-selected k), then name
each cluster by interpreting its centroid's *raw* features against football
archetypes (depth, width, involvement, pressing, attack/defence position
delta). The taxonomy comes from the data; the names come from the rules.

Also flags players whose discovered role family disagrees with their nominal
formation slot (e.g. a nominal fullback clustering with midfielders — the
inverted-fullback signature).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitchiq.config import RolesConfig
from pitchiq.core.formations import LABEL_FAMILY
from pitchiq.intelligence.embeddings import EmbeddingResult
from pitchiq.intelligence.features import PlayerFeatures

ROLE_FAMILY = {
    "ball-playing centre-back": "centre-back",
    "no-nonsense centre-back": "centre-back",
    "attacking fullback": "fullback",
    "defensive fullback": "fullback",
    "deep-lying playmaker": "defensive-mid",
    "ball-winning midfielder": "defensive-mid",
    "box-to-box midfielder": "central-mid",
    "advanced playmaker": "attacking-mid",
    "wide midfielder": "wide-mid",
    "winger": "winger",
    "pressing forward": "striker",
    "target striker": "striker",
    "mobile forward": "striker",
}


def discover_roles(emb: EmbeddingResult, features: dict[int, PlayerFeatures],
                   cfg: RolesConfig) -> dict:
    """Cluster embeddings and name the clusters. Returns the roles artifact."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = emb.vectors
    n = len(emb.ids)
    if n < 6:
        return {"players": {}, "clusters": [], "note": "too few players for role discovery"}

    if cfg.n_clusters == "auto":
        best_k, best_s = 2, -1.0
        for k in range(4, min(9, n - 1) + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
            if len(set(km.labels_)) < 2:
                continue
            s = silhouette_score(X, km.labels_)
            if s > best_s:
                best_k, best_s = k, s
        k = best_k
    else:
        k = int(cfg.n_clusters)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    labels = km.labels_

    clusters = []
    role_of_cluster: dict[int, str] = {}
    for c in range(k):
        members = [emb.ids[i] for i in range(n) if labels[i] == c]
        name, traits = _name_cluster(members, features)
        role_of_cluster[c] = name
        clusters.append({"cluster": c, "role": name, "members": members, "traits": traits})

    players = {}
    for i, eid in enumerate(emb.ids):
        players[str(eid)] = {
            "role": role_of_cluster[int(labels[i])],
            "cluster": int(labels[i]),
            "team": features[eid].team,
        }
    return {"players": players, "clusters": clusters, "k": k,
            "embedding_method": emb.method}


def _name_cluster(members: list[int], features: dict[int, PlayerFeatures]) -> tuple[str, dict]:
    """Interpret a cluster's centroid features into a role name."""
    def mean_of(group: str, key: str, default=0.0) -> float:
        vals = [features[m].groups.get(group, {}).get(key) for m in members]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else default

    depth = mean_of("phase", "def_mean_x", mean_of("spatial", "mean_x", 40.0))
    att_x = mean_of("phase", "att_mean_x", depth)
    push_up = mean_of("phase", "push_up_delta")
    wide = mean_of("spatial", "wideness", 12.0)
    touches = mean_of("involvement", "touches_per_min", 0.5)
    press = mean_of("involvement", "press_per_min", 0.3)
    dist = mean_of("movement", "dist_per_min", 90.0)
    centrality = mean_of("interaction", "net_betweenness", 0.0)
    x_range = mean_of("spatial", "x_range", 20.0)

    traits = dict(def_depth_m=round(depth, 1), att_depth_m=round(att_x, 1),
                  push_up_m=round(push_up, 1), wideness_m=round(wide, 1),
                  touches_per_min=round(touches, 2), press_per_min=round(press, 2),
                  dist_per_min=round(dist, 1), betweenness=round(centrality, 3),
                  x_range_m=round(x_range, 1))

    central = wide < 13.0
    deep = depth < 32.0
    mid = 32.0 <= depth < 52.0
    high = depth >= 52.0
    busy = touches > np.float64(1.2)
    pressy = press > 0.9
    runner = dist > 105.0 or x_range > 32.0

    if deep and central:
        name = "ball-playing centre-back" if busy or centrality > 0.05 else "no-nonsense centre-back"
    elif deep and not central:
        name = "attacking fullback" if push_up > 9.0 or runner else "defensive fullback"
    elif mid and central:
        if busy and depth < 42.0:
            name = "deep-lying playmaker"
        elif pressy and not busy:
            name = "ball-winning midfielder"
        elif runner:
            name = "box-to-box midfielder"
        elif busy:
            name = "advanced playmaker"
        else:
            name = "box-to-box midfielder"
    elif mid and not central:
        name = "wide midfielder"
    elif high and not central:
        name = "winger"
    else:  # high & central
        if pressy:
            name = "pressing forward"
        elif runner:
            name = "mobile forward"
        else:
            name = "target striker"
    return name, traits


def nominal_vs_actual(roles: dict, formation_windows: pd.DataFrame) -> list[dict]:
    """Players whose discovered role family ≠ nominal formation-slot family."""
    if not len(formation_windows):
        return []
    # nominal slot: modal label across windows where the player appears
    slot_votes: dict[int, list[str]] = {}
    for _, w in formation_windows.iterrows():
        for pid, slot in zip(w["players"], w["slots"]):
            slot_votes.setdefault(int(pid), []).append(slot)
    mismatches = []
    for eid_str, info in roles.get("players", {}).items():
        eid = int(eid_str)
        votes = slot_votes.get(eid)
        if not votes:
            continue
        nominal = max(set(votes), key=votes.count)
        nominal_family = LABEL_FAMILY.get(nominal, "?")
        actual_family = ROLE_FAMILY.get(info["role"], "?")
        if nominal_family != actual_family and "?" not in (nominal_family, actual_family):
            mismatches.append({
                "entity_id": eid, "team": info["team"],
                "nominal_slot": nominal, "nominal_family": nominal_family,
                "discovered_role": info["role"], "role_family": actual_family,
            })
    return mismatches
