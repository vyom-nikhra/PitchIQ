"""Pass networks (graph + centrality) and line-breaking passes."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from pitchiq.core.schema import MatchMeta
from pitchiq.analytics.common import attack_sign_series, to_attacking_x


def build_pass_network(events: pd.DataFrame, df: pd.DataFrame, team: str) -> dict:
    """Pass-network for one team: nodes at average pass-involvement position,
    edges weighted by completed pass volume, plus centrality metrics."""
    passes = events[(events.type == "pass") & (events.team == team)
                    & (events.outcome == "complete") & (events.to_id >= 0)]
    if not len(passes):
        return {"nodes": [], "edges": [], "metrics": {}}

    G = nx.DiGraph()
    for _, p in passes.iterrows():
        u, v = int(p.from_id), int(p.to_id)
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)

    # node positions: mean of pass origins (as passer) and receptions
    pos_acc: dict[int, list] = {}
    for _, p in passes.iterrows():
        pos_acc.setdefault(int(p.from_id), []).append((p.x, p.y))
        pos_acc.setdefault(int(p.to_id), []).append((p.end_x, p.end_y))
    node_pos = {k: np.nanmean(np.array(v), axis=0) for k, v in pos_acc.items()}

    degree = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight=lambda u, v, d: 1.0 / d["weight"])
    try:
        eigen = nx.eigenvector_centrality(G.to_undirected(), weight="weight", max_iter=500)
    except nx.PowerIterationFailedConvergence:
        eigen = {n: 0.0 for n in G}

    nodes = [
        dict(id=int(n), x=round(float(node_pos[n][0]), 1), y=round(float(node_pos[n][1]), 1),
             volume=int(degree.get(n, 0)),
             betweenness=round(float(betweenness.get(n, 0)), 4),
             eigenvector=round(float(eigen.get(n, 0)), 4))
        for n in G.nodes if n in node_pos
    ]
    edges = [dict(source=int(u), target=int(v), weight=int(d["weight"]))
             for u, v, d in G.edges(data=True)]
    top_pair = max(edges, key=lambda e: e["weight"]) if edges else None
    metrics = {
        "n_passes": int(len(passes)),
        "n_players": int(G.number_of_nodes()),
        "density": round(float(nx.density(G)), 3),
        "top_combination": top_pair,
        "most_central": max(nodes, key=lambda n: n["betweenness"])["id"] if nodes else None,
    }
    return {"nodes": nodes, "edges": edges, "metrics": metrics}


def defensive_lines_at(df: pd.DataFrame, frame: int, defending_team: str,
                       meta: MatchMeta, tolerance: int = 3) -> list[float]:
    """The defending team's line depths (attacking-x of the POSSESSING team's
    view) at a frame: outfielders clustered into up to 3 lines by depth gaps."""
    near = df[(df.frame >= frame - tolerance) & (df.frame <= frame + tolerance)
              & (df["class"] == "player") & (df.team == defending_team)]
    if near.empty:
        return []
    pos = near.groupby("entity_id")[["x_pitch"]].mean()
    attacking_team = "home" if defending_team == "away" else "away"
    sign = attack_sign_series(meta, np.array([frame]), attacking_team)[0]
    x_att = np.sort(to_attacking_x(pos.x_pitch.to_numpy(), np.full(len(pos), sign),
                                   meta.pitch_length))
    if len(x_att) < 6:
        return []
    # split at the two largest depth gaps -> up to 3 lines
    gaps = np.diff(x_att)
    split_idx = np.sort(np.argsort(gaps)[-2:]) + 1
    lines = np.split(x_att, split_idx)
    return [float(np.mean(l)) for l in lines if len(l) >= 2]


def line_breaking_passes(events: pd.DataFrame, df: pd.DataFrame, meta: MatchMeta) -> pd.DataFrame:
    """Completed passes that cross a defensive line in the attacking direction.

    A pass breaks a line when its start is at least 2m behind the line depth
    and its end at least 2m beyond it (both in the attacking team's frame).
    """
    rows = []
    passes = events[(events.type == "pass") & (events.outcome == "complete")]
    for _, p in passes.iterrows():
        team = str(p.team)
        defending = "away" if team == "home" else "home"
        lines = defensive_lines_at(df, int(p.frame), defending, meta)
        if not lines:
            continue
        sign = attack_sign_series(meta, np.array([int(p.frame)]), team)[0]
        x0 = to_attacking_x(np.array([p.x]), np.array([sign]), meta.pitch_length)[0]
        x1 = to_attacking_x(np.array([p.end_x]), np.array([sign]), meta.pitch_length)[0]
        # measured in the ATTACKING team's frame the defender line depths are
        # distances from the attacking goal... convert: defending team lines were
        # computed in attacking team frame already (see defensive_lines_at).
        broken = [ln for ln in lines if x0 < ln - 2.0 and x1 > ln + 2.0]
        if broken:
            rows.append(dict(frame=int(p.frame), team=team, from_id=int(p.from_id),
                             to_id=int(p.to_id), n_lines_broken=len(broken),
                             x=float(p.x), y=float(p.y),
                             end_x=float(p.end_x), end_y=float(p.end_y)))
    return pd.DataFrame(rows)
