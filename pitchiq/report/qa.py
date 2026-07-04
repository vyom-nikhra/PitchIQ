"""Match Q&A: questions answered strictly from the computed facts.

LLM path: the facts JSON rides along with every question under a grounding
prompt. Fallback path: keyword retrieval over the flattened facts —
term-overlap scoring plus a small synonym map, returning the matching metrics
verbatim. Both paths refuse rather than invent when nothing matches.
"""

from __future__ import annotations

import json
import re

from pitchiq.config import ReportConfig
from pitchiq.report import llm as llm_mod
from pitchiq.report.facts import flatten_facts

QA_SYSTEM = """You answer questions about one football match using ONLY the \
provided JSON metrics. Quote the numbers you use. If the answer is not in the \
JSON, reply exactly: "The extracted analytics don't cover that." Keep answers \
to 2-5 sentences, analyst tone."""

SYNONYMS = {
    "press": ["pressing", "ppda", "pressure", "pressures"],
    "possession": ["ball share", "share"],
    "formation": ["formations", "shape", "morph"],
    "mark": ["marking", "man", "zonal", "scheme", "pairs"],
    "fast": ["speed", "top_speed", "sprints"],
    "run": ["distance", "sprints", "physical"],
    "threat": ["xt", "expected_threat", "created"],
    "pass": ["passes", "pass_network", "line_breaking", "combination"],
    "role": ["roles", "discovered", "nominal"],
    "tilt": ["field_tilt", "territory"],
    "counter": ["counter_attacks", "transition"],
    "block": ["shape", "def_line_height_m", "low_block", "high"],
}


def answer_question(question: str, facts: dict, cfg: ReportConfig) -> dict:
    """Returns {'answer': str, 'engine': 'gemini'|'anthropic'|'retrieval',
    'evidence': [dotted keys]}."""
    provider = llm_mod.resolve_provider(cfg.llm)
    evidence = _retrieve(question, facts, k=8)
    if provider != "none":
        user = (
            "Match metrics JSON:\n```json\n"
            + json.dumps(facts, indent=1, default=str)[:24000]
            + f"\n```\n\nQuestion: {question}"
        )
        text = llm_mod.generate(QA_SYSTEM, user, cfg.llm)
        if text:
            return {"answer": text.strip(), "engine": provider,
                    "evidence": [k for k, _ in evidence]}
    # deterministic retrieval fallback
    if not evidence:
        return {"answer": "The extracted analytics don't cover that.",
                "engine": "retrieval", "evidence": []}
    lines = [f"- `{k}` = {v}" for k, v in evidence]
    return {"answer": "Closest computed metrics:\n" + "\n".join(lines),
            "engine": "retrieval", "evidence": [k for k, _ in evidence]}


def _retrieve(question: str, facts: dict, k: int = 8) -> list[tuple[str, str]]:
    corpus = flatten_facts(facts)
    terms = set(re.findall(r"[a-z]+", question.lower()))
    expanded = set(terms)
    for t in terms:
        for key, syns in SYNONYMS.items():
            if t.startswith(key) or key.startswith(t):
                expanded.update(s.replace(" ", "_") for s in syns)
                expanded.add(key)
    scored = []
    for key, val in corpus:
        key_terms = set(re.findall(r"[a-z]+", key.lower()))
        overlap = len(expanded & key_terms)
        if overlap:
            scored.append((overlap, key, val))
    scored.sort(key=lambda t: -t[0])
    return [(key, val) for _, key, val in scored[:k]]
