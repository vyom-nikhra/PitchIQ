"""Video IO: frame reading with fps resampling, and H.264-capable writing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)


class FrameReader:
    """Iterate video frames as ``(frame_idx, timestamp_s, bgr_image)``.

    ``target_fps`` strides frames down to approximately that rate (never
    upsamples); ``frame_idx`` counts *emitted* frames and timestamps stay true
    to the source clock, so the tracking table is internally consistent at the
    processing rate.
    """

    def __init__(
        self,
        path: str | Path,
        target_fps: float | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        self.src_fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.src_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.stride = 1
        if target_fps and target_fps < self.src_fps:
            self.stride = max(1, round(self.src_fps / target_fps))
        self.fps = self.src_fps / self.stride
        self.max_frames = max_frames

    @property
    def n_frames_estimate(self) -> int:
        n = self.src_frame_count // self.stride if self.src_frame_count > 0 else 0
        if self.max_frames:
            n = min(n, self.max_frames) if n else self.max_frames
        return n

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        emitted = 0
        src_idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if src_idx % self.stride == 0:
                yield emitted, src_idx / self.src_fps, frame
                emitted += 1
                if self.max_frames and emitted >= self.max_frames:
                    break
            src_idx += 1
        self.cap.release()

    def close(self) -> None:
        self.cap.release()


class VideoSink:
    """Write BGR frames to mp4. Prefers H.264 via imageio-ffmpeg (plays in
    browsers); falls back to OpenCV's mp4v with a logged warning."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.size = size  # (width, height)
        self._writer = None
        self._ffmpeg_proc = None
        try:
            import imageio_ffmpeg

            self._ffmpeg_proc = imageio_ffmpeg.write_frames(
                str(self.path),
                size,
                fps=fps,
                codec="libx264",
                quality=7,
                pix_fmt_in="bgr24",
                output_params=["-movflags", "+faststart"],
            )
            self._ffmpeg_proc.send(None)  # prime the generator
        except Exception as exc:  # pragma: no cover - environment dependent
            log.warning("imageio-ffmpeg unavailable (%s); falling back to mp4v", exc)
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
            )

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[1] != self.size[0] or frame_bgr.shape[0] != self.size[1]:
            frame_bgr = cv2.resize(frame_bgr, self.size)
        if self._ffmpeg_proc is not None:
            self._ffmpeg_proc.send(np.ascontiguousarray(frame_bgr))
        else:
            self._writer.write(frame_bgr)

    def close(self) -> None:
        if self._ffmpeg_proc is not None:
            self._ffmpeg_proc.close()
        if self._writer is not None:
            self._writer.release()

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
