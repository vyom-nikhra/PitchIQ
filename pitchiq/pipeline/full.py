"""End-to-end orchestration: video (or tracking table) → every artifact.

Stage weights feed one continuous progress bar; every stage writes through
``store.update_status`` so the app can poll. Heavy CV runs once — analytics /
intelligence / report are pure re-runs on the cached parquet.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

from pitchiq.config import Config
from pitchiq.core.artifacts import ArtifactStore

log = logging.getLogger(__name__)

STAGES_VIDEO = [("perception", 0.0, 0.55), ("analytics", 0.55, 0.72),
                ("intelligence", 0.72, 0.82), ("report", 0.82, 0.88),
                ("media", 0.88, 1.0)]
STAGES_DATA = [("analytics", 0.0, 0.45), ("intelligence", 0.45, 0.7),
               ("report", 0.7, 0.8), ("media", 0.8, 1.0)]


class FullPipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ api
    def process_video(self, video_path: str | Path, store: ArtifactStore) -> None:
        """Raw clip → full dashboard artifacts."""
        self._run(store, video_path=Path(video_path), stages=STAGES_VIDEO)

    def process_tracking_only(self, store: ArtifactStore,
                              video_path: str | Path | None = None) -> None:
        """Tracking table already in the store (simulator / Metrica import)."""
        self._run(store, video_path=Path(video_path) if video_path else None,
                  stages=STAGES_DATA)

    # ------------------------------------------------------------------ impl
    def _run(self, store: ArtifactStore, video_path: Path | None, stages) -> None:
        span = {name: (a, b) for name, a, b in stages}

        def cb(stage: str):
            a, b = span[stage]

            def _cb(p: float, msg: str) -> None:
                store.update_status(stage, a + p * (b - a), msg)

            return _cb

        try:
            store.update_status("queued", 0.0, "starting", state="running")
            if "perception" in span:
                from pitchiq.pipeline.perception import PerceptionPipeline

                PerceptionPipeline(self.cfg).run(video_path, store, cb("perception"))

            from pitchiq.pipeline.analytics import AnalyticsPipeline

            AnalyticsPipeline(self.cfg).run(store, cb("analytics"))

            from pitchiq.pipeline.intelligence import IntelligencePipeline

            IntelligencePipeline(self.cfg).run(store, cb("intelligence"))

            from pitchiq.pipeline.report import ReportPipeline

            ReportPipeline(self.cfg).run(store, cb("report"))

            self._media(store, video_path, cb("media"))
            store.update_status("done", 1.0, "complete", state="done")
        except Exception as exc:
            log.error("pipeline failed: %s\n%s", exc, traceback.format_exc())
            store.update_status("error", 1.0, f"{type(exc).__name__}: {exc}",
                                state="error")
            raise

    def _media(self, store: ArtifactStore, video_path: Path | None, progress) -> None:
        """Annotated video + small preview encode for the radar component."""
        if video_path is None or not Path(video_path).exists():
            progress(1.0, "no source video — radar runs standalone")
            return
        from pitchiq.viz.annotate import annotate_video

        tracking = store.load_tracking()
        meta = store.load_meta()
        progress(0.05, "annotating video")
        annotated = annotate_video(video_path, tracking, meta,
                                   store.media_path("annotated.mp4"),
                                   progress_cb=lambda p, m: progress(0.05 + 0.75 * p, m))
        progress(0.82, "encoding radar preview")
        make_preview(annotated, store.media_path("preview.mp4"), width=640)
        progress(1.0, "media complete")


def make_preview(src: str | Path, dst: str | Path, width: int = 640) -> Path:
    """Small H.264 re-encode for browser embedding (radar sync component)."""
    import cv2

    from pitchiq.core.video import FrameReader, VideoSink

    reader = FrameReader(src)
    # multiple of 16 keeps ffmpeg from macro-block resizing (which would
    # subtly rescale pixels relative to any overlay math)
    height = max(16, int(round(reader.height * width / reader.width / 16) * 16))
    with VideoSink(dst, reader.fps, (width, height)) as sink:
        for _idx, _ts, frame in reader:
            sink.write(cv2.resize(frame, (width, height)))
    return Path(dst)
