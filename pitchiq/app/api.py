"""FastAPI backend: async job processing over the artifact store.

Endpoints:
    GET  /health
    GET  /api/jobs                    list all jobs (+ bundled demo matches)
    POST /api/jobs                    upload a clip → async processing
    GET  /api/jobs/{job_id}/status
    GET  /api/jobs/{job_id}/artifacts/{path}   serve any artifact file
    POST /api/jobs/{job_id}/qa        grounded match Q&A

Processing runs on a single worker thread guarded by a semaphore — one heavy
CV job at a time; further uploads queue. The registry is stateless: jobs are
whatever directories exist under the artifacts root, so restarts lose nothing.
(A Celery/RQ worker pool is the documented production upgrade.)
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from pitchiq.config import load_config
from pitchiq.core.artifacts import ArtifactStore
from pitchiq.core.env import load_env

log = logging.getLogger(__name__)
load_env()

cfg = load_config()
JOBS_ROOT = Path(cfg.app.artifacts_root)
DEMO_ROOT = Path(cfg.app.demo_root)
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PitchIQ API", version="0.1.0")
_worker_gate = threading.Semaphore(1)


def _job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", job_id):
        raise HTTPException(400, "invalid job id")
    for root in (JOBS_ROOT, DEMO_ROOT):
        p = root / job_id
        if p.exists():
            return p
    raise HTTPException(404, f"unknown job {job_id}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "pitchiq"}


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    out = []
    for root, kind in ((DEMO_ROOT, "demo"), (JOBS_ROOT, "upload")):
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            store = ArtifactStore(d)
            status = store.read_status()
            name = d.name
            if store.meta_path.exists():
                meta = store.load_meta()
                name = f"{meta.team_names.get('home')} vs {meta.team_names.get('away')}"
            out.append({"job_id": d.name, "name": name, "kind": kind,
                        "state": status.get("state", "unknown"),
                        "progress": status.get("progress", 0.0)})
    return out


@app.post("/api/jobs")
async def submit_job(file: UploadFile) -> dict:
    if not (file.filename or "").lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
        raise HTTPException(400, "upload a video file (.mp4/.mov/.mkv/.avi)")
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    store = ArtifactStore(JOBS_ROOT / job_id)
    with open(store.input_video, "wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
    store.update_status("queued", 0.0, "waiting for worker", state="queued")

    def run() -> None:
        from pitchiq.pipeline.full import FullPipeline

        with _worker_gate:
            try:
                FullPipeline(load_config()).process_video(store.input_video, store)
            except Exception:
                log.exception("job %s failed", job_id)

    threading.Thread(target=run, daemon=True, name=f"job-{job_id}").start()
    return {"job_id": job_id, "state": "queued"}


@app.get("/api/jobs/{job_id}/status")
def job_status(job_id: str) -> dict:
    return ArtifactStore(_job_dir(job_id)).read_status()


@app.get("/api/jobs/{job_id}/artifacts/{artifact_path:path}")
def get_artifact(job_id: str, artifact_path: str):
    root = _job_dir(job_id).resolve()
    target = (root / artifact_path).resolve()
    if not str(target).startswith(str(root)):  # path-traversal guard
        raise HTTPException(403, "forbidden")
    if not target.is_file():
        raise HTTPException(404, "artifact not found")
    return FileResponse(target)


class Question(BaseModel):
    question: str


@app.post("/api/jobs/{job_id}/qa")
def job_qa(job_id: str, q: Question) -> JSONResponse:
    from pitchiq.report.qa import answer_question

    store = ArtifactStore(_job_dir(job_id))
    facts_path = store.report_path("facts.json")
    if not facts_path.exists():
        raise HTTPException(409, "report facts not ready yet")
    facts = store.load_json(facts_path)
    return JSONResponse(answer_question(q.question, facts, cfg.report))
