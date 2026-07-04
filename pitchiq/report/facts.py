"""Assemble the grounded fact base for report generation and Q&A.

Everything the LLM is allowed to say comes from this dict — computed numbers
with stable keys and human-readable player labels. It is persisted as
``report/facts.json`` so every sentence in the report can be traced back to a
metric, and so Q&A retrieval works without re-running analytics.
"""

from __future__ import annotations

import pandas as pd

from pitchiq.core.artifacts import ArtifactStore


def _player_label(eid: int | str, players: dict, team_names: dict) -> str:
    p = players.get(str(eid), {})
    j = p.get("jersey_no")
    team = team_names.get(p.get("team", ""), p.get("team", "?"))
    core = f"#{int(j)}" if j is not None else f"player {eid}"
    return f"{core} ({team})"


def build_facts(store: ArtifactStore) -> dict:
    """Collect analytics + intelligence outputs into one grounded fact base."""
    meta = store.load_meta()
    summary = store.load_json(store.analytics_path("summary.json"))
    intel_path = store.intelligence_path("summary.json")
    intel = store.load_json(intel_path) if intel_path.exists() else {}
    roles_path = store.intelligence_path("roles.json")
    roles = store.load_json(roles_path) if roles_path.exists() else {}
    marking_path = store.intelligence_path("marking.json")
    marking = store.load_json(marking_path) if marking_path.exists() else {}

    players = summary.get("players", {})
    team_names = meta.team_names
    label = lambda eid: _player_label(eid, players, team_names)  # noqa: E731

    minutes = meta.n_frames / meta.fps / 60.0
    facts: dict = {
        "match": {
            "teams": team_names,
            "duration_min": round(minutes, 1),
            "source": meta.source,
            "is_synthetic": "synthetic" in (meta.source or ""),
        },
        "possession": summary.get("possession", {}),
        "field_tilt": summary.get("field_tilt", {}),
        "territory": summary.get("territory", {}),
        "formations": summary.get("formations", {}),
        "shape": summary.get("shape", {}),
        "pitch_control": summary.get("pitch_control", {}),
        "pressing": summary.get("pressing", {}),
        "ppda": summary.get("ppda", {}),
        "events": {k: v for k, v in summary.get("events", {}).items()
                   if k != "counter_attacks"},
        "counter_attacks": summary.get("events", {}).get("counter_attacks", []),
        "line_breaking": summary.get("line_breaking", {}),
        "phases": summary.get("phases", {}),
        "team_distance_m": summary.get("team_distance_m", {}),
    }

    # top movers / speedsters with labels (team players only — referees are
    # tracked for completeness but are not report material)
    profs = [(eid, p) for eid, p in players.items()
             if p.get("minutes", 0) > 1 and p.get("team") in ("home", "away")]
    by_dist = sorted(profs, key=lambda kv: -kv[1].get("distance_per_min_m", 0))[:5]
    by_speed = sorted(profs, key=lambda kv: -kv[1].get("top_speed_mps", 0))[:5]
    facts["physical"] = {
        "top_distance_per_min": [
            {"player": label(e), "value_m_per_min": p["distance_per_min_m"],
             "sprints": p["n_sprints"]} for e, p in by_dist],
        "top_speed": [
            {"player": label(e), "value_mps": p["top_speed_mps"]} for e, p in by_speed],
    }

    # xT with labels (round away float32 representation noise)
    xt = summary.get("xt", {})
    facts["expected_threat"] = {
        "top_players": [
            {"player": label(r["entity_id"]), "xt_created": round(float(r["xt_created"]), 4),
             "n_moves": int(r["n_moves"])} for r in xt.get("top_players", [])],
        "team_created": {k: round(float(v), 4)
                         for k, v in xt.get("team_created", {}).items()},
    }

    # pass network key facts with labels
    pn = summary.get("pass_network", {})
    facts["pass_network"] = {}
    for team, net in pn.items():
        m = net.get("metrics", {})
        entry = {"n_passes": m.get("n_passes"), "density": m.get("density")}
        if m.get("most_central") is not None:
            entry["most_central_player"] = label(m["most_central"])
        tp = m.get("top_combination")
        if tp:
            entry["top_combination"] = (
                f"{label(tp['source'])} → {label(tp['target'])} ({tp['weight']} passes)")
        facts["pass_network"][team] = entry

    # roles + mismatches with labels
    if roles:
        facts["roles"] = {
            "by_player": {label(e): info.get("role")
                          for e, info in roles.get("players", {}).items()},
            "clusters": [{"role": c["role"], "members": [label(m) for m in c["members"]]}
                         for c in roles.get("clusters", [])],
            "nominal_vs_actual": [
                {"player": label(m["entity_id"]), "nominal": m["nominal_slot"],
                 "discovered": m["discovered_role"]}
                for m in roles.get("nominal_vs_actual", [])],
            "embedding_method": roles.get("embedding_method"),
        }

    # marking with labels
    if marking:
        facts["marking"] = {}
        for team, entry in marking.items():
            op = entry.get("open_play", {})
            if "team_man_score" not in op:
                continue
            facts["marking"][team] = {
                "scheme": op.get("scheme"),
                "man_score": op.get("team_man_score"),
                "pairs": [
                    {"defender": label(p["defender_id"]),
                     "marks": label(p["attacker_id"]),
                     "share": p["share"]} for p in op.get("pairs", [])[:6]],
            }
            sp = entry.get("set_piece", {})
            if "team_man_score" in sp:
                facts["marking"][team]["set_piece_scheme"] = sp.get("scheme")

    if intel:
        facts["intelligence_meta"] = {
            "n_players_embedded": intel.get("n_players_embedded"),
            "embedding_method": intel.get("embedding_method"),
        }
    return facts


def flatten_facts(facts: dict, prefix: str = "") -> list[tuple[str, str]]:
    """(dotted_key, value_string) pairs — the retrieval corpus for Q&A."""
    out: list[tuple[str, str]] = []
    if isinstance(facts, dict):
        for k, v in facts.items():
            out.extend(flatten_facts(v, f"{prefix}{k}." if prefix or True else k))
    elif isinstance(facts, list):
        for i, v in enumerate(facts[:12]):
            out.extend(flatten_facts(v, f"{prefix}{i}."))
    else:
        out.append((prefix.rstrip("."), str(facts)))
    return out
