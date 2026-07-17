"""Intelligence-layer tests: features, embeddings, roles, similarity, marking."""

import numpy as np
import pandas as pd

from pitchiq.config import EmbeddingsConfig, MarkingConfig, RolesConfig, SimilarityConfig
from pitchiq.intelligence.embeddings import compute_handcrafted
from pitchiq.intelligence.features import PlayerFeatures
from pitchiq.intelligence.marking import analyse_marking
from pitchiq.intelligence.roles import _name_cluster, discover_roles
from pitchiq.intelligence.similarity import SimilarityIndex

FPS = 25.0


# ---------------------------------------------------------------- fixtures
def _fake_features(n_per_archetype=4, seed=0) -> dict[int, PlayerFeatures]:
    """Three synthetic archetypes with distinct feature signatures."""
    rng = np.random.default_rng(seed)
    archetypes = {
        # (def_x, att_x, wideness, touches, press, dist)
        "cb": (20, 30, 8, 0.8, 0.3, 85),
        "b2b": (40, 60, 10, 1.5, 0.8, 115),
        "winger": (55, 75, 24, 1.1, 0.6, 100),
    }
    out = {}
    eid = 1
    for name, (dx, ax, wide, touch, press, dist) in archetypes.items():
        for _ in range(n_per_archetype):
            hm = np.zeros((16, 24), dtype=np.float32)
            cx = int((dx + ax) / 2 / 105 * 24)
            cy = int((34 + (wide if eid % 2 else -wide)) / 68 * 16) % 16
            hm[max(0, cy - 1): cy + 2, max(0, cx - 1): cx + 2] = 1.0
            hm /= hm.sum()
            pf = PlayerFeatures(entity_id=eid, team="home" if eid % 2 else "away",
                                minutes=45.0)
            j = rng.normal(0, 0.05)
            pf.groups = {
                "spatial": {"mean_x": (dx + ax) / 2 + rng.normal(0, 2), "mean_y": 34.0,
                            "std_x": 8.0, "std_y": 6.0, "wideness": wide + rng.normal(0, 1),
                            "hull_area": 300.0, "x_range": ax - dx + 10, "y_range": 20.0},
                "movement": {"dist_per_min": dist * (1 + j), "top_speed": 8.0,
                             "speed_p50": 1.5, "speed_p90": 5.0, "sprints_per_min": 0.4,
                             "hi_dist_per_min": 8.0, "accel_p90": 2.0,
                             "dir_changes_per_min": 3.0},
                "involvement": {"touches_per_min": touch + rng.normal(0, 0.1),
                                "poss_time_share": 0.05, "mean_dist_ball": 25.0,
                                "near_ball_share": 0.3, "press_per_min": press,
                                "time_defending": 0.4},
                "interaction": {"dist_to_centroid": 12.0, "net_volume": 2.0,
                                "net_betweenness": 0.1, "net_eigenvector": 0.3},
                "phase": {"att_mean_x": ax + rng.normal(0, 2), "att_mean_y": 34.0,
                          "def_mean_x": dx + rng.normal(0, 2), "def_mean_y": 34.0,
                          "push_up_delta": ax - dx, "tuck_in_delta": 0.0},
            }
            pf.heatmap = hm
            out[eid] = pf
            eid += 1
    return out


def test_handcrafted_embedding_clusters_archetypes():
    feats = _fake_features()
    emb = compute_handcrafted(feats, EmbeddingsConfig())
    assert emb.vectors.shape[0] == 12
    assert np.allclose(np.linalg.norm(emb.vectors, axis=1), 1.0, atol=1e-5)
    # same-archetype similarity must exceed cross-archetype similarity
    V = emb.vectors
    ids = emb.ids
    group_of = {eid: (eid - 1) // 4 for eid in ids}
    same, cross = [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            (same if group_of[ids[i]] == group_of[ids[j]] else cross).append(
                float(V[i] @ V[j]))
    assert np.mean(same) > np.mean(cross) + 0.3


def test_role_discovery_finds_archetypes_and_names():
    feats = _fake_features()
    emb = compute_handcrafted(feats, EmbeddingsConfig())
    roles = discover_roles(emb, feats, RolesConfig(n_clusters=3))
    assert roles["k"] == 3
    # members of each cluster come from one archetype
    for c in roles["clusters"]:
        arch = {(m - 1) // 4 for m in c["members"]}
        assert len(arch) == 1, f"cluster mixes archetypes: {c['members']}"
    names = {c["role"] for c in roles["clusters"]}
    assert any("centre-back" in n for n in names)


def test_cluster_naming_rules():
    feats = _fake_features()
    name_cb, traits = _name_cluster([1, 2, 3, 4], feats)
    assert "centre-back" in name_cb
    name_w, _ = _name_cluster([9, 10, 11, 12], feats)
    assert name_w in ("winger", "wide midfielder", "mobile forward", "pressing forward")


def test_similarity_index_and_attribution():
    feats = _fake_features()
    emb = compute_handcrafted(feats, EmbeddingsConfig())
    idx = SimilarityIndex(emb, SimilarityConfig(top_k=3))
    hits = idx.query(1)  # a CB
    assert hits[0]["entity_id"] in (2, 3, 4)  # nearest is another CB
    assert 0 <= hits[0]["similarity"] <= 1.0001
    assert set(hits[0]["drivers"]) == {"involvement", "interaction", "movement",
                                       "phase", "spatial"}


# ---------------------------------------------------------------- marking
def _marking_fixture(scheme: str, n_frames=1500, seed=3, n_players=6):
    """NvN defensive scenario. Attackers rotate positions slowly; 'man'
    defenders shadow their attacker at 2m; 'zonal' defenders hold fixed zones."""
    rng = np.random.default_rng(seed)
    n = n_players
    lanes = np.linspace(8, 62, n)
    zones = np.column_stack([np.full(n, 30.0), lanes])
    kin_rows, phase_rows = [], []
    for f in range(n_frames):
        t = f / FPS
        # attackers rotate through each other's lanes + individual wiggle
        att = np.column_stack([np.full(n, 45.0), lanes])
        att[:, 1] = 34 + 26 * np.sin(2 * np.pi * (t / 60) + np.arange(n) * 2 * np.pi / n)
        att[:, 0] += 4 * np.sin(2 * np.pi * t / 9 + np.arange(n))
        if scheme == "man":
            defs = att + np.array([-2.5, 0.0]) + rng.normal(0, 0.15, (n, 2))
        else:
            defs = zones + rng.normal(0, 0.6, (n, 2))
            # zonal block still shifts gently with the attack's centroid
            defs[:, 1] += 0.25 * (att[:, 1].mean() - 34)
        for i in range(n):
            kin_rows.append(dict(frame=f, entity_id=100 + i, x=att[i, 0], y=att[i, 1],
                                 vx=0.0, vy=0.0, speed=1.0, accel=0.0))
            kin_rows.append(dict(frame=f, entity_id=200 + i, x=defs[i, 0], y=defs[i, 1],
                                 vx=0.0, vy=0.0, speed=1.0, accel=0.0))
        phase_rows.append(dict(frame=f, poss_team="home", phase="progression",
                               def_posture="mid_block"))
    kin = pd.DataFrame(kin_rows)
    phases = pd.DataFrame(phase_rows)
    team_of = {100 + i: "home" for i in range(n)} | {200 + i: "away" for i in range(n)}
    return kin, phases, team_of


def test_marking_man_vs_zonal_separation():
    cfg = MarkingConfig(min_defensive_frames=20)
    kin_m, ph, team_of = _marking_fixture("man")
    man = analyse_marking(kin_m, ph, team_of, cfg, FPS)["away"]["open_play"]
    kin_z, ph2, team_of2 = _marking_fixture("zonal")
    zonal = analyse_marking(kin_z, ph2, team_of2, cfg, FPS)["away"]["open_play"]
    assert man["team_man_score"] > 0.72, man
    assert zonal["team_man_score"] < 0.5, zonal
    assert man["scheme"] == "man-marking"
    assert zonal["scheme"] == "zonal"
    # who-marks-whom: shadowing defenders recover their attacker exactly
    pairs = {p["defender_id"]: p["attacker_id"] for p in man["pairs"]}
    correct = sum(1 for d, a in pairs.items() if a == d - 100)
    assert correct >= 5


def test_marking_excludes_goalkeepers():
    cfg = MarkingConfig(min_defensive_frames=20)
    kin, ph, team_of = _marking_fixture("man")
    class_of = {eid: ("goalkeeper" if eid == 200 else "player") for eid in team_of}
    res = analyse_marking(kin, ph, team_of, cfg, FPS, class_of=class_of)
    pairs = res["away"]["open_play"]["pairs"]
    assert all(p["defender_id"] != 200 for p in pairs)
