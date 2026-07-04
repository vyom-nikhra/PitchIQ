"""Report-stage orchestration: facts → report.md (+ generator provenance)."""

from __future__ import annotations

import logging
from typing import Callable

from pitchiq.config import Config
from pitchiq.core.artifacts import ArtifactStore
from pitchiq.report.facts import build_facts
from pitchiq.report.generator import build_report

log = logging.getLogger(__name__)


class ReportPipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self, store: ArtifactStore,
            progress_cb: Callable[[float, str], None] | None = None) -> dict:
        if progress_cb:
            progress_cb(0.1, "assembling grounded facts")
        facts = build_facts(store)
        store.save_json(store.report_path("facts.json"), facts)
        if progress_cb:
            progress_cb(0.4, "generating analyst report")
        report_md, engine = build_report(facts, self.cfg.report)
        store.report_path("report.md").write_text(report_md, encoding="utf-8")
        store.save_json(store.report_path("report_meta.json"),
                        {"generator": engine})
        log.info("report generated via %s (%d chars)", engine, len(report_md))
        if progress_cb:
            progress_cb(1.0, f"report complete ({engine})")
        return {"generator": engine, "chars": len(report_md)}
