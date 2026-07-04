"""Viz smoke tests: payloads and figures build without a browser."""

import numpy as np
import pandas as pd

from pitchiq.core.schema import BALL_ID, MatchMeta
from pitchiq.viz.charts import heatmap_fig, pass_network_fig, pitch_control_fig
from pitchiq.viz.pitch_plot import pitch_figure
from pitchiq.viz.radar_html import build_radar_html, radar_payload


def _mini_tracking(n_frames=50, fps=25.0):
    rows = []
    for f in range(n_frames):
        for eid, team in ((1, "home"), (2, "away")):
            rows.append(dict(frame=f, timestamp=f / fps, entity_id=eid,
                             **{"class": "player"}, team=team, jersey_no=eid * 7,
                             x_pixel=np.nan, y_pixel=np.nan,
                             x_pitch=30.0 + eid * 10 + 0.1 * f, y_pitch=34.0,
                             conf=1.0))
        rows.append(dict(frame=f, timestamp=f / fps, entity_id=BALL_ID,
                         **{"class": "ball"}, team="none", jersey_no=None,
                         x_pixel=np.nan, y_pixel=np.nan,
                         x_pitch=50.0, y_pitch=34.0, conf=1.0))
    return pd.DataFrame(rows)


def test_radar_payload_resampling():
    meta = MatchMeta(fps=25, n_frames=50)
    payload = radar_payload(_mini_tracking(), meta, radar_fps=12.5)
    assert abs(payload["fps"] - 12.5) < 1e-6
    assert len(payload["frames"]) == 25  # stride 2
    assert len(payload["ball"]) == 25
    assert payload["ball"][0] == [50.0, 34.0]
    ents = payload["frames"][0]
    assert {e[0] for e in ents} == {1, 2}
    assert payload["labels"]["1"] == "7" if "1" in payload["labels"] else payload["labels"][1] == "7"


def test_radar_html_standalone():
    meta = MatchMeta(fps=25, n_frames=50)
    html = build_radar_html(_mini_tracking(), meta, video_path=None)
    assert "<canvas" in html and "requestAnimationFrame" in html
    assert 'id="seek"' in html  # standalone controls when no video
    assert "drawFrame" in html


def test_figures_build():
    assert len(pitch_figure().layout.shapes) > 10
    hm = heatmap_fig(np.random.rand(34, 52).astype(np.float32), "test")
    assert hm.data
    pc = pitch_control_fig(np.random.rand(34, 52).astype(np.float32),
                           {"home": "#d62728", "away": "#1f77b4"})
    assert pc.data
    net = {"nodes": [{"id": 1, "x": 30, "y": 30, "volume": 5, "betweenness": 0.1,
                      "eigenvector": 0.2},
                     {"id": 2, "x": 50, "y": 40, "volume": 3, "betweenness": 0.0,
                      "eigenvector": 0.1}],
           "edges": [{"source": 1, "target": 2, "weight": 4}], "metrics": {}}
    pn = pass_network_fig(net, "#d62728", "Reds")
    assert len(pn.data) >= 2
