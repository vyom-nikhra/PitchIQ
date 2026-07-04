"""Analyst charts (plotly) consumed by the Streamlit dashboard.

Each function takes artifacts (arrays/frames/dicts) and returns a Figure —
no file IO here, so charts are testable and reusable from notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pitchiq.core.pitch import Pitch
from pitchiq.viz.pitch_plot import pitch_figure


def heatmap_fig(grid: np.ndarray, title: str, pitch: Pitch | None = None) -> go.Figure:
    pitch = pitch or Pitch()
    fig = pitch_figure(pitch)
    ny, nx = grid.shape
    fig.add_trace(go.Heatmap(
        z=grid,
        x=np.linspace(0, pitch.length, nx),
        y=np.linspace(0, pitch.width, ny),
        colorscale=[[0, "rgba(0,0,0,0)"], [0.25, "rgba(255,220,0,0.35)"],
                    [0.6, "rgba(255,120,0,0.6)"], [1, "rgba(220,0,0,0.85)"]],
        showscale=False, hoverinfo="skip"))
    fig.update_layout(title=dict(text=title, font=dict(size=14)))
    return fig


def pitch_control_fig(mean_home_control: np.ndarray, kit: dict[str, str],
                      pitch: Pitch | None = None) -> go.Figure:
    pitch = pitch or Pitch()
    fig = pitch_figure(pitch)
    ny, nx = mean_home_control.shape
    fig.add_trace(go.Heatmap(
        z=mean_home_control,
        x=np.linspace(0, pitch.length, nx),
        y=np.linspace(0, pitch.width, ny),
        zmin=0, zmax=1, colorscale=[[0, kit.get("away", "#1f77b4")],
                                    [0.5, "rgba(255,255,255,0.15)"],
                                    [1, kit.get("home", "#d62728")]],
        opacity=0.62, colorbar=dict(title="home control", thickness=10)))
    fig.update_layout(title=dict(text="Average pitch control", font=dict(size=14)))
    return fig


def pass_network_fig(net: dict, kit_color: str, team_name: str,
                     pitch: Pitch | None = None) -> go.Figure:
    fig = pitch_figure(pitch)
    nodes = {n["id"]: n for n in net.get("nodes", [])}
    max_w = max((e["weight"] for e in net.get("edges", [])), default=1)
    for e in net.get("edges", []):
        a, b = nodes.get(e["source"]), nodes.get(e["target"])
        if not a or not b:
            continue
        fig.add_trace(go.Scatter(
            x=[a["x"], b["x"]], y=[a["y"], b["y"]], mode="lines",
            line=dict(color="rgba(255,255,255,0.75)",
                      width=0.8 + 4.5 * e["weight"] / max_w),
            hoverinfo="text", text=f"{e['source']}→{e['target']}: {e['weight']}",
            showlegend=False))
    if nodes:
        xs = [n["x"] for n in nodes.values()]
        ys = [n["y"] for n in nodes.values()]
        sizes = [12 + 1.6 * n["volume"] for n in nodes.values()]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=sizes, color=kit_color,
                        line=dict(color="white", width=1.5)),
            text=[str(i) for i in nodes], textposition="middle center",
            textfont=dict(color="white", size=10),
            hovertext=[f"id {i}: volume {n['volume']}, betweenness {n['betweenness']}"
                       for i, n in nodes.items()],
            hoverinfo="text", name=team_name))
    fig.update_layout(title=dict(text=f"Pass network — {team_name}", font=dict(size=14)))
    return fig


def marking_fig(timeline: pd.DataFrame, kin: pd.DataFrame, frame: int,
                kit: dict[str, str], defending_team: str,
                pitch: Pitch | None = None) -> go.Figure:
    """Marking lines at (nearest sampled) frame: defender → assigned attacker."""
    fig = pitch_figure(pitch)
    tl = timeline[timeline.defending_team == defending_team]
    if not len(tl):
        return fig
    frames = np.sort(tl.frame.unique())
    f = int(frames[np.argmin(np.abs(frames - frame))])
    snap = tl[tl.frame == f]
    pos = kin[kin.frame == f].set_index("entity_id")[["x", "y"]]
    def_color = kit.get(defending_team, "#d62728")
    att_color = kit.get("away" if defending_team == "home" else "home", "#1f77b4")
    for _, row in snap.iterrows():
        try:
            d = pos.loc[int(row.defender_id)]
            a = pos.loc[int(row.attacker_id)]
        except KeyError:
            continue
        fig.add_trace(go.Scatter(x=[d.x, a.x], y=[d.y, a.y], mode="lines",
                                 line=dict(color="rgba(255,255,60,0.9)", width=1.6,
                                           dash="dot"),
                                 hoverinfo="text",
                                 text=f"{int(row.defender_id)} marks {int(row.attacker_id)} "
                                      f"({row.dist_m} m)", showlegend=False))
        fig.add_trace(go.Scatter(x=[d.x], y=[d.y], mode="markers",
                                 marker=dict(size=13, color=def_color,
                                             line=dict(color="white", width=1)),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[a.x], y=[a.y], mode="markers",
                                 marker=dict(size=13, color=att_color,
                                             line=dict(color="white", width=1)),
                                 showlegend=False, hoverinfo="skip"))
    fig.update_layout(title=dict(
        text=f"Marking assignments — {defending_team} defending (frame {f})",
        font=dict(size=14)))
    return fig


def xt_bar_fig(xt_players: pd.DataFrame, labels: dict[int, str],
               kit: dict[str, str]) -> go.Figure:
    df = xt_players.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=df["xt_created"], y=[labels.get(int(e), str(e)) for e in df["entity_id"]],
        orientation="h",
        marker_color=[kit.get(t, "#888") for t in df["team"]],
    ))
    fig.update_layout(title="Expected-threat creation (xT)", height=380,
                      margin=dict(l=8, r=8, t=40, b=8),
                      xaxis_title="xT created")
    return fig


def shape_timeline_fig(shape_ts: pd.DataFrame, kit: dict[str, str], fps: float) -> go.Figure:
    fig = go.Figure()
    for team, g in shape_ts.groupby("team"):
        g = g.sort_values("frame")
        fig.add_trace(go.Scatter(
            x=g.frame / fps / 60.0, y=g.def_line_height_m, mode="lines",
            name=f"{team} def. line", line=dict(color=kit.get(str(team), "#888"))))
    fig.update_layout(title="Defensive line height over time", height=320,
                      xaxis_title="minute", yaxis_title="line height (m from own goal)",
                      margin=dict(l=8, r=8, t=40, b=8))
    return fig


def phase_share_fig(phase_summary: dict, team_names: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    for team in ("home", "away"):
        shares = phase_summary.get(f"{team}_in_possession", {})
        if shares:
            fig.add_trace(go.Bar(name=team_names.get(team, team),
                                 x=list(shares.keys()), y=list(shares.values())))
    fig.update_layout(barmode="group", title="Phase shares in possession",
                      height=320, margin=dict(l=8, r=8, t=40, b=8),
                      yaxis_title="share of possession time")
    return fig


def possession_flow_fig(possession: pd.DataFrame, kit: dict[str, str],
                        fps: float, window_s: float = 30.0) -> go.Figure:
    p = possession.copy()
    p["is_home"] = (p.team == "home").astype(float)
    p["is_contested"] = (p.team == "none").astype(float)
    w = max(1, int(window_s * fps))
    roll = p.set_index("frame")["is_home"].rolling(w, min_periods=w // 3).mean()
    fig = go.Figure(go.Scatter(x=roll.index / fps / 60.0, y=roll, mode="lines",
                               line=dict(color=kit.get("home", "#d62728")),
                               fill="tozeroy", name="home share"))
    fig.add_hline(y=0.5, line=dict(color="rgba(255,255,255,0.4)", dash="dot"))
    fig.update_layout(title=f"Rolling possession ({window_s:.0f}s window)", height=300,
                      xaxis_title="minute", yaxis=dict(range=[0, 1], title="home share"),
                      margin=dict(l=8, r=8, t=40, b=8))
    return fig
