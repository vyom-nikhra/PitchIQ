"""Annotated video: boxes, IDs, team colours, jersey numbers over the source clip.

Reads the tracking table (pixel coordinates) and redraws it onto the video —
a pure post-processing pass, so it reruns instantly on cached tables. Player
markers use broadcast-style foot ellipses rather than raw rectangles; the
ball gets a trailing marker. An optional mini-radar is composited in a corner.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pitchiq.core.pitch import Pitch
from pitchiq.core.schema import BALL_ID, MatchMeta
from pitchiq.core.video import FrameReader, VideoSink

TEAM_FALLBACK = {"home": (40, 40, 220), "away": (220, 120, 30), "none": (60, 220, 220)}


def _hex_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i: i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def annotate_video(
    video_path: str | Path,
    tracking: pd.DataFrame,
    meta: MatchMeta,
    out_path: str | Path,
    with_radar: bool = True,
    progress_cb=None,
) -> Path:
    colors = {k: _hex_bgr(v) for k, v in meta.kit_colors.items()}
    colors.setdefault("none", TEAM_FALLBACK["none"])
    by_frame = dict(tuple(tracking.groupby("frame")))
    pitch = Pitch(meta.pitch_length, meta.pitch_width)
    radar_bg = _radar_background(pitch) if with_radar else None

    reader = FrameReader(video_path, target_fps=meta.fps)
    n = reader.n_frames_estimate or 1
    out_path = Path(out_path)
    with VideoSink(out_path, reader.fps, (reader.width, reader.height)) as sink:
        for idx, ts, frame in reader:
            rows = by_frame.get(idx)
            if rows is not None:
                _draw_frame(frame, rows, colors)
                if radar_bg is not None:
                    _composite_radar(frame, radar_bg, rows, colors, pitch)
            sink.write(frame)
            if progress_cb and idx % 100 == 0:
                progress_cb(idx / n, f"annotating frame {idx}/{n}")
    return out_path


def _draw_frame(frame: np.ndarray, rows: pd.DataFrame, colors: dict) -> None:
    for _, r in rows.iterrows():
        if not np.isfinite(r.x_pixel) or not np.isfinite(r.y_pixel):
            continue
        x, y = int(r.x_pixel), int(r.y_pixel)
        if r.entity_id == BALL_ID:
            cv2.circle(frame, (x, y), 7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 2, (0, 215, 255), -1, cv2.LINE_AA)
            continue
        color = colors.get(str(r.team), TEAM_FALLBACK.get(str(r.team), (200, 200, 200)))
        if r["class"] == "referee":
            color = (30, 30, 30)
        elif r["class"] == "goalkeeper":
            color = tuple(min(255, c + 70) for c in color)
        cv2.ellipse(frame, (x, y), (14, 6), 0, -30, 210, color, 2, cv2.LINE_AA)
        label = str(int(r.jersey_no)) if pd.notna(r.jersey_no) else str(int(r.entity_id))
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        cv2.rectangle(frame, (x - tw // 2 - 3, y + 8), (x + tw // 2 + 3, y + 22),
                      color, -1)
        cv2.putText(frame, label, (x - tw // 2, y + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def _radar_background(pitch: Pitch, w: int = 260) -> np.ndarray:
    h = int(w * pitch.width / pitch.length)
    img = np.full((h, w, 3), (35, 90, 40), dtype=np.uint8)
    sx, sy = w / pitch.length, h / pitch.width
    for (x1, y1), (x2, y2) in pitch.lines.values():
        cv2.line(img, (int(x1 * sx), h - int(y1 * sy)), (int(x2 * sx), h - int(y2 * sy)),
                 (230, 230, 230), 1, cv2.LINE_AA)
    c = pitch.circles["center"]
    cv2.circle(img, (int(c.cx * sx), h - int(c.cy * sy)), int(c.r * sx),
               (230, 230, 230), 1, cv2.LINE_AA)
    return img


def _composite_radar(frame: np.ndarray, radar_bg: np.ndarray, rows: pd.DataFrame,
                     colors: dict, pitch: Pitch) -> None:
    radar = radar_bg.copy()
    h, w = radar.shape[:2]
    sx, sy = w / pitch.length, h / pitch.width
    for _, r in rows.iterrows():
        if not np.isfinite(r.x_pitch) or not np.isfinite(r.y_pitch):
            continue
        x, y = int(r.x_pitch * sx), h - int(r.y_pitch * sy)
        if r.entity_id == BALL_ID:
            cv2.circle(radar, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            color = colors.get(str(r.team), (200, 200, 200))
            cv2.circle(radar, (x, y), 4, color, -1, cv2.LINE_AA)
    fh, fw = frame.shape[:2]
    pad = 12
    y0 = fh - h - pad
    x0 = fw - w - pad
    roi = frame[y0: y0 + h, x0: x0 + w]
    cv2.addWeighted(radar, 0.85, roi, 0.15, 0, dst=roi)
    cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + w, y0 + h), (255, 255, 255), 1)
