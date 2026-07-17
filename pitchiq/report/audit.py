"""Groundedness audit: verify a report's numeric claims against facts.json.

The report layer promises "every quantitative claim traces to facts.json";
this measures that promise. Numbers are extracted from the report narrative
(the Metrics Appendix is excluded — it *is* the facts) and matched against
every numeric token in the fact base, tolerating printed rounding and
fraction<->percent conversion. Formation codes (4-3-3) and jersey refs (#7)
are not treated as standalone numbers.
"""

from __future__ import annotations

import re

from pitchiq.report.facts import flatten_facts

FORMATION = re.compile(r"\b\d(?:-\d){1,3}\b")
NUMBER = re.compile(r"(?<![\w#.\d-])(\d+(?:\.\d+)?)")


def fact_numbers(facts: dict) -> set[float]:
    """Every numeric token in the fact base (values may be labels like
    '#7 (Reds)' — their digits count as grounded ids/jerseys)."""
    out: set[float] = set()
    for _key, value in flatten_facts(facts):
        for tok in NUMBER.findall(FORMATION.sub(" ", value)):
            try:
                out.add(float(tok))
            except ValueError:
                continue
    return out


def matches(claim: float, decimals: int, pool: set[float]) -> bool:
    """True when some fact value equals the claim at its printed precision,
    directly or as a fraction printed as a percentage."""
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(claim - v) <= tol or abs(claim - 100 * v) <= tol
               for v in pool)


def audit(report_md: str, facts: dict) -> tuple[list[tuple[str, str, bool]], float]:
    """Returns ([(number_text, context, grounded)], grounded_share)."""
    body = report_md.split("## Metrics Appendix")[0]
    pool = fact_numbers(facts)
    results: list[tuple[str, str, bool]] = []
    scrubbed = FORMATION.sub(" ", body)
    for m in NUMBER.finditer(scrubbed):
        tok = m.group(1)
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        ctx = scrubbed[max(0, m.start() - 35):m.end() + 25].replace("\n", " ")
        results.append((tok, ctx.strip(), matches(float(tok), decimals, pool)))
    grounded = sum(1 for _, _, ok in results if ok)
    share = grounded / len(results) if results else 1.0
    return results, share
