"""The live tactical map: a top-down radar synced to video scrubbing.

Builds a self-contained HTML component (embedded in Streamlit via
``components.html``; portable to any React shell later): a <video> element
and a <canvas> radar drawn from an embedded JSON of resampled positions,
synchronised on every animation frame from ``video.currentTime`` — scrubbing,
pausing and playback speed all stay perfectly in sync.

The clip is base64-embedded (kept small by the demo build's 640px preview
encode). If no/oversized video, the radar runs standalone with its own
play/scrub controls, so the component always works.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitchiq.core.pitch import Pitch
from pitchiq.core.schema import BALL_ID, MatchMeta

TEAM_CODE = {"home": 0, "away": 1, "none": 2}


def radar_payload(tracking: pd.DataFrame, meta: MatchMeta, radar_fps: float) -> dict:
    """Compact position payload resampled to ``radar_fps``."""
    stride = max(1, int(round(meta.fps / radar_fps)))
    eff_fps = meta.fps / stride
    frames_wanted = np.arange(0, int(tracking.frame.max()) + 1, stride)
    sub = tracking[tracking.frame.isin(frames_wanted)]
    persons = sub[sub["class"] != "ball"].dropna(subset=["x_pitch", "y_pitch"])
    ball = sub[sub.entity_id == BALL_ID].set_index("frame")

    frames: list[list] = []
    balls: list = []
    by_frame = dict(tuple(persons.groupby("frame")))
    labels: dict[int, str] = {}
    for f in frames_wanted:
        ents = []
        g = by_frame.get(f)
        if g is not None:
            for _, r in g.iterrows():
                eid = int(r.entity_id)
                ents.append([eid, TEAM_CODE.get(str(r.team), 2),
                             round(float(r.x_pitch), 1), round(float(r.y_pitch), 1)])
                if eid not in labels:
                    labels[eid] = (str(int(r.jersey_no)) if pd.notna(r.jersey_no)
                                   else str(eid))
        frames.append(ents)
        if f in ball.index and np.isfinite(ball.loc[f, "x_pitch"]):
            balls.append([round(float(ball.loc[f, "x_pitch"]), 1),
                          round(float(ball.loc[f, "y_pitch"]), 1)])
        else:
            balls.append(None)
    return {
        "fps": eff_fps,
        "L": meta.pitch_length,
        "W": meta.pitch_width,
        "colors": {"0": meta.kit_colors.get("home", "#d62728"),
                   "1": meta.kit_colors.get("away", "#1f77b4"), "2": "#f5e642"},
        "names": {"0": meta.team_names.get("home", "Home"),
                  "1": meta.team_names.get("away", "Away")},
        "labels": labels,
        "frames": frames,
        "ball": balls,
    }


def build_radar_html(tracking: pd.DataFrame, meta: MatchMeta,
                     video_path: str | Path | None = None,
                     radar_fps: float = 12.5, max_video_mb: float = 40.0) -> str:
    payload = radar_payload(tracking, meta, radar_fps)
    video_tag = ""
    has_video = False
    if video_path and Path(video_path).exists():
        size_mb = Path(video_path).stat().st_size / 1e6
        if size_mb <= max_video_mb:
            b64 = base64.b64encode(Path(video_path).read_bytes()).decode()
            video_tag = (f'<video id="vid" controls preload="auto" '
                         f'src="data:video/mp4;base64,{b64}"></video>')
            has_video = True

    pitch = Pitch(meta.pitch_length, meta.pitch_width)
    lines = [[*a, *b] for a, b in pitch.lines.values()]
    data_js = json.dumps(payload, separators=(",", ":"))
    lines_js = json.dumps(lines)

    controls = "" if has_video else (
        '<div id="ctl"><button id="play">▶</button>'
        '<input id="seek" type="range" min="0" max="1000" value="0" style="flex:1"></div>')

    return f"""
<style>
  .piq-wrap {{ display:flex; flex-direction:column; gap:8px;
               font-family: 'Segoe UI', sans-serif; }}
  .piq-wrap video {{ width:100%; border-radius:8px; background:#000; }}
  .piq-wrap canvas {{ width:100%; border-radius:8px; }}
  #ctl {{ display:flex; gap:8px; align-items:center; }}
  #ctl button {{ width:44px; height:30px; border-radius:6px; border:none;
                 background:#2e7d32; color:#fff; cursor:pointer; }}
  .clock {{ color:#ddd; font-size:12px; text-align:right; }}
</style>
<div class="piq-wrap">
  {video_tag}
  <canvas id="radar" width="900" height="{int(900 * (pitch.width + 6) / (pitch.length + 6))}"></canvas>
  {controls}
  <div class="clock" id="clock">0:00.0</div>
</div>
<script>
const D = {data_js};
const LINES = {lines_js};
const cv = document.getElementById('radar');
const ctx = cv.getContext('2d');
const vid = document.getElementById('vid');
const M = 3;  // metre margin
const sx = cv.width / (D.L + 2*M), sy = cv.height / (D.W + 2*M);
const X = x => (x + M) * sx, Y = y => cv.height - (y + M) * sy;
let t = 0, playing = false, lastTs = null;

function drawPitch() {{
  ctx.fillStyle = '#14501f'; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = '#1d6f2b'; ctx.fillRect(X(0), Y(D.W), D.L*sx, D.W*sy);
  ctx.strokeStyle = 'rgba(255,255,255,.8)'; ctx.lineWidth = 1.4;
  for (const [x1,y1,x2,y2] of LINES) {{
    ctx.beginPath(); ctx.moveTo(X(x1), Y(y1)); ctx.lineTo(X(x2), Y(y2)); ctx.stroke();
  }}
  ctx.beginPath(); ctx.arc(X(D.L/2), Y(D.W/2), 9.15*sx, 0, 7); ctx.stroke();
}}

function drawFrame(fi) {{
  drawPitch();
  const ents = D.frames[fi] || [];
  for (const [id, tc, x, y] of ents) {{
    ctx.beginPath(); ctx.arc(X(x), Y(y), tc === 2 ? 5 : 8, 0, 7);
    ctx.fillStyle = D.colors[tc]; ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,.9)'; ctx.lineWidth = 1; ctx.stroke();
    if (tc !== 2) {{
      ctx.fillStyle = '#fff'; ctx.font = 'bold 9px sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(D.labels[id] || id, X(x), Y(y));
    }}
  }}
  const b = D.ball[fi];
  if (b) {{
    ctx.beginPath(); ctx.arc(X(b[0]), Y(b[1]), 4.5, 0, 7);
    ctx.fillStyle = '#fff'; ctx.fill();
    ctx.strokeStyle = '#222'; ctx.stroke();
  }}
  // legend
  ctx.font = '12px sans-serif'; ctx.textAlign = 'left';
  ctx.fillStyle = D.colors[0]; ctx.fillRect(10, 8, 12, 12);
  ctx.fillStyle = '#fff'; ctx.fillText(D.names[0], 26, 17);
  ctx.fillStyle = D.colors[1]; ctx.fillRect(120, 8, 12, 12);
  ctx.fillStyle = '#fff'; ctx.fillText(D.names[1], 136, 17);
  const s = fi / D.fps;
  document.getElementById('clock').textContent =
    Math.floor(s/60) + ':' + (s % 60).toFixed(1).padStart(4, '0');
}}

function loop(ts) {{
  if (vid) {{
    t = vid.currentTime;
  }} else if (playing) {{
    if (lastTs !== null) t += (ts - lastTs) / 1000;
    lastTs = ts;
    const seek = document.getElementById('seek');
    if (seek) seek.value = Math.round(1000 * t * D.fps / D.frames.length);
  }}
  let fi = Math.min(D.frames.length - 1, Math.max(0, Math.round(t * D.fps)));
  drawFrame(fi);
  requestAnimationFrame(loop);
}}
if (!vid) {{
  const btn = document.getElementById('play'), seek = document.getElementById('seek');
  btn.onclick = () => {{ playing = !playing; lastTs = null;
                         btn.textContent = playing ? '⏸' : '▶'; }};
  seek.oninput = () => {{ t = seek.value / 1000 * D.frames.length / D.fps; }};
}}
requestAnimationFrame(loop);
</script>
"""
