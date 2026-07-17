"""Report tests: facts assembly, template generation, Q&A retrieval.

LLM calls are NOT exercised here (network); provider resolution is tested
with keys absent so CI is deterministic.
"""

import numpy as np
import pytest

from pitchiq.config import LLMConfig, ReportConfig
from pitchiq.core.artifacts import ArtifactStore
from pitchiq.core.schema import MatchMeta
from pitchiq.report.facts import build_facts, flatten_facts
from pitchiq.report.generator import build_report, template_report
from pitchiq.report.qa import answer_question


@pytest.fixture()
def mini_store(tmp_path) -> ArtifactStore:
    store = ArtifactStore(tmp_path / "job")
    MatchMeta(fps=25, n_frames=1500, source="synthetic-simulator",
              team_names={"home": "Reds", "away": "Blues"}).save(store.meta_path)
    store.save_json(store.analytics_path("summary.json"), {
        "players": {
            "1": {"team": "home", "jersey_no": 7, "minutes": 4.0,
                  "distance_per_min_m": 110.0, "n_sprints": 5, "top_speed_mps": 8.1},
            "200": {"team": "none", "jersey_no": None, "minutes": 4.0,
                    "distance_per_min_m": 300.0, "n_sprints": 0, "top_speed_mps": 6.0},
        },
        "possession": {"share": {"home": 0.61, "away": 0.39}},
        "field_tilt": {"tilt_home": 0.7,
                       "home": {"final_third_share": 0.4, "own_third_share": 0.2,
                                "mean_ball_x_att": 60.0}},
        "formations": {"home": {"in_possession": {"formation": "4-3-3", "avg_width_m": 45.0},
                                "out_possession": {"formation": "4-5-1"},
                                "shape_morph": "4-3-3 in possession → 4-5-1 out of possession"}},
        "shape": {"home": {"block_label": "high line",
                           "out_possession": {"def_line_height_m": 42.0, "depth_m": 25.0}}},
        "pressing": {"home": {"pressures": 30, "pressures_per_min": 7.5,
                              "press_height_mean_m": 60.0, "press_to_turnover_rate": 0.4}},
        "ppda": {"home": {"ppda": 6.0}, "away": {"ppda": 14.0}},
        "events": {"n_passes": 80, "n_completed_passes": 66, "n_turnovers": 20,
                   "n_carries": 12, "counter_attacks": []},
        "line_breaking": {"total": 9, "by_team": {"home": 6, "away": 3}},
        "xt": {"top_players": [{"entity_id": 1, "xt_created": np.float32(0.05),
                                "n_moves": 9}], "team_created": {"home": 0.09}},
        "pass_network": {"home": {"metrics": {"n_passes": 50, "density": 0.4,
                                              "most_central": 1,
                                              "top_combination": {"source": 1, "target": 1,
                                                                  "weight": 7}}}},
        "phases": {}, "territory": {}, "team_distance_m": {"home": 5000, "away": 4800},
    })
    return store


def test_build_facts_labels_and_filters(mini_store):
    facts = build_facts(mini_store)
    assert facts["match"]["is_synthetic"] is True
    labels = [r["player"] for r in facts["physical"]["top_distance_per_min"]]
    assert "#7 (Reds)" in labels[0]
    # referee (team none) excluded from physical facts
    assert not any("200" in l for l in labels)
    assert facts["expected_threat"]["top_players"][0]["xt_created"] == 0.05
    # perception-quality assessment always rides along (ground-truth here)
    assert facts["data_quality"]["overall"] == "high"
    assert facts["data_quality"]["is_cv"] is False


def test_template_report_contains_grounded_numbers(mini_store):
    facts = build_facts(mini_store)
    md = template_report(facts)
    assert "61%" in md
    assert "4-3-3" in md and "4-5-1" in md
    assert "PPDA 6.0" in md
    assert "#7 (Reds)" in md
    assert "Simulated demonstration" in md
    assert "Data confidence: **high**" in md


def test_template_report_surfaces_low_confidence():
    facts = {"match": {"teams": {"home": "H", "away": "A"}},
             "possession": {"share": {"home": 0.6}},
             "data_quality": {"overall": "low", "is_cv": True,
                              "notes": ["The ball was directly observed in "
                                        "only 12% of frames."]}}
    md = template_report(facts)
    assert "Data confidence: **low**" in md
    assert "12% of frames" in md  # note appears in exec summary and watch-outs


def test_build_report_falls_back_without_keys(mini_store, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import pitchiq.core.env as envmod

    monkeypatch.setattr(envmod, "load_env", lambda path=None: None)
    facts = build_facts(mini_store)
    md, engine = build_report(facts, ReportConfig(llm=LLMConfig(provider="auto")))
    assert engine == "template"
    assert "## Metrics Appendix" in md


def test_qa_retrieval_fallback(mini_store, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import pitchiq.core.env as envmod

    monkeypatch.setattr(envmod, "load_env", lambda path=None: None)
    facts = build_facts(mini_store)
    ans = answer_question("How intense was the home pressing?", facts,
                          ReportConfig(llm=LLMConfig(provider="none")))
    assert ans["engine"] == "retrieval"
    assert any("ppda" in e or "pressing" in e for e in ans["evidence"])
    nonsense = answer_question("What colour were the corner flags?", facts,
                               ReportConfig(llm=LLMConfig(provider="none")))
    assert "don't cover" in nonsense["answer"] or nonsense["evidence"] == [] \
        or len(nonsense["evidence"]) <= 8


def test_flatten_facts_roundtrip():
    flat = flatten_facts({"a": {"b": 1, "c": [{"d": 2}]}})
    keys = dict(flat)
    assert keys["a.b"] == "1"
    assert keys["a.c.0.d"] == "2"
