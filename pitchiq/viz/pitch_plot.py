"""Plotly pitch figure builder — the canvas for every tactical chart."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from pitchiq.core.pitch import Pitch

PITCH_GREEN = "#1d6f2b"
LINE_COLOR = "rgba(255,255,255,0.85)"


def pitch_figure(pitch: Pitch | None = None, height: int = 480,
                 bg: str = PITCH_GREEN) -> go.Figure:
    """A styled, correctly-proportioned pitch with all markings."""
    pitch = pitch or Pitch()
    fig = go.Figure()
    shapes = [
        dict(type="rect", x0=-2, y0=-2, x1=pitch.length + 2, y1=pitch.width + 2,
             fillcolor=bg, line=dict(width=0), layer="below"),
    ]
    for (x1, y1), (x2, y2) in pitch.lines.values():
        shapes.append(dict(type="line", x0=x1, y0=y1, x1=x2, y1=y2,
                           line=dict(color=LINE_COLOR, width=1.6), layer="below"))
    for c in pitch.circles.values():
        t0, t1 = c.theta_range if c.theta_range else (0, 2 * np.pi)
        th = np.linspace(t0, t1, 60)
        fig.add_trace(go.Scatter(
            x=c.cx + c.r * np.cos(th), y=c.cy + c.r * np.sin(th),
            mode="lines", line=dict(color=LINE_COLOR, width=1.6),
            hoverinfo="skip", showlegend=False))
    for name in ("penalty_spot_left", "penalty_spot_right", "center_spot"):
        x, y = pitch.keypoints[name]
        shapes.append(dict(type="circle", x0=x - 0.4, y0=y - 0.4, x1=x + 0.4, y1=y + 0.4,
                           fillcolor=LINE_COLOR, line=dict(width=0), layer="below"))
    fig.update_layout(
        shapes=shapes,
        xaxis=dict(range=[-3, pitch.length + 3], visible=False, fixedrange=True),
        yaxis=dict(range=[-3, pitch.width + 3], visible=False, fixedrange=True,
                   scaleanchor="x", scaleratio=1),
        height=height, margin=dict(l=8, r=8, t=28, b=8),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0.0,
                    font=dict(size=11)),
    )
    return fig
