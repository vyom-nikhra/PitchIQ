"""Quantitative validation against synthetic ground truth → docs/validation.md.

Renders a fresh simulated broadcast clip, runs the REAL perception pipeline
on it, and scores every layer against the simulator/renderer ground truth:

* calibration — positional error of GT boxes projected through estimated H
* tracking    — MOTA / IDF1 / ID switches (foot-point boxes, IoU proxy)
* possession  — frame agreement with the simulator's possession sequence
* passes      — precision/recall of derived passes vs the GT pass log
* marking     — recovered man-marking pairs + man/zonal score separation

Synthetic validation is necessary, not sufficient — real-broadcast caveats
are documented in the README. Runtime ~10 min CPU (dominated by perception).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def foot_boxes(df: pd.DataFrame, w: float = 30.0, h: float = 60.0) -> pd.DataFrame:
    """Fixed-size boxes centred on foot pixels — lets the IoU-based MOT
    metrics act as a distance proxy when only points are available."""
    out = df.dropna(subset=["x_pixel", "y_pixel"]).copy()
    out["x1"] = out.x_pixel - w / 2
    out["x2"] = out.x_pixel + w / 2
    out["y1"] = out.y_pixel - h
    out["y2"] = out.y_pixel
    out["track_id"] = out.entity_id
    return out[["frame", "track_id", "x1", "y1", "x2", "y2"]]


def main() -> None:
    from pitchiq.config import load_config
    from pitchiq.core.artifacts import ArtifactStore
    from pitchiq.core.geometry import apply_homography
    from pitchiq.core.pitch import Pitch
    from pitchiq.core.schema import homography_for_frame
    from pitchiq.demo.render import render_match
    from pitchiq.demo.simulate import simulate_demo_match
    from pitchiq.perception.tracking.metrics import evaluate_tracking
    from pitchiq.pipeline.perception import PerceptionPipeline
    from pitchiq.pipeline.analytics import AnalyticsPipeline
    from pitchiq.pipeline.intelligence import IntelligencePipeline

    out_dir = REPO / "data" / "jobs" / "_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(out_dir)

    cfg = load_config(overrides={
        "simulator": {"half_minutes": 1.0, "seed": 21},
        "detection": {"backend": "blob"},
        "jersey": {"enabled": False},
    })
    print("simulating + rendering ground-truth clip...")
    sim = simulate_demo_match(cfg.simulator)
    rr = render_match(sim.tracking, Pitch(), sim.meta.kit_colors,
                      store.input_video, cfg.simulator.fps)

    print("running perception pipeline (this is the slow part)...")
    t0 = time.time()
    pred, hdf, meta = PerceptionPipeline(cfg).run(store.input_video, store)
    percep_s = time.time() - t0

    # ------------------------------------------------------- calibration
    gt_pos = sim.tracking
    cal_errs = []
    for f, g in rr.boxes[rr.boxes["class"] == "player"].groupby("frame"):
        H = homography_for_frame(hdf, int(f))
        if H is None:
            continue
        m = g.merge(gt_pos[gt_pos.frame == f][["entity_id", "x_pitch", "y_pitch"]],
                    on="entity_id")
        if not len(m):
            continue
        feet = np.stack([(m.x1 + m.x2) / 2, m.y2], axis=1)
        proj = apply_homography(H, feet)
        cal_errs.append(np.linalg.norm(proj - m[["x_pitch", "y_pitch"]].to_numpy(), axis=1))
    cal = np.concatenate(cal_errs)
    calibrated_frac = float(hdf.dropna(subset=["h00"]).shape[0] / max(len(hdf), 1))

    # ---------------------------------------------------------- tracking
    gt_boxes = rr.boxes[rr.boxes["class"].isin(["player", "goalkeeper"])].copy()
    gt_boxes["track_id"] = gt_boxes.entity_id
    mot = evaluate_tracking(
        gt_boxes[["frame", "track_id", "x1", "y1", "x2", "y2"]],
        foot_boxes(pred[pred["class"].isin(["player", "goalkeeper"])]),
        iou_thresh=0.2,  # loose: proxy boxes have arbitrary fixed size
    )

    # -------------------------------------------------------- analytics
    print("running analytics + intelligence on CV output...")
    AnalyticsPipeline(cfg).run(store)
    IntelligencePipeline(cfg).run(store)
    poss = pd.read_parquet(store.analytics_path("possession.parquet"))
    poss_pred = poss.set_index("frame")["team"].reindex(
        range(len(sim.possession_gt))).fillna("none").to_numpy(dtype=object)
    agree = float((poss_pred == sim.possession_gt).mean())

    ev = store.load_events()
    der_p = ev[ev.type == "pass"]
    gt_p = sim.events[sim.events.type == "pass"]
    used, matched = set(), 0
    for _, d in der_p.iterrows():
        cand = gt_p[(np.abs(gt_p.frame - d.frame) < 40)]
        for gi in cand.index:
            if gi not in used:
                used.add(gi)
                matched += 1
                break
    pass_prec = matched / max(len(der_p), 1)
    pass_rec = matched / max(len(gt_p), 1)

    marking = store.load_json(store.intelligence_path("marking.json"))

    # ------------------------------------------------------------ report
    md = f"""# Synthetic validation report

Auto-generated by `scripts/validate_synthetic.py` — the full CV pipeline
(blob detector variant) run on a rendered ground-truth broadcast clip
({sim.meta.n_frames} frames @ {sim.meta.fps:.0f} fps; perception {percep_s:.0f}s CPU).

| Layer | Metric | Value |
|---|---|---|
| Calibration | frames with homography | {100 * calibrated_frac:.1f}% |
| Calibration | positional error mean | {cal.mean():.2f} m |
| Calibration | positional error median | {np.median(cal):.2f} m |
| Calibration | positional error p90 | {np.percentile(cal, 90):.2f} m |
| Tracking | MOTA (foot-box proxy) | {mot['mota']:.3f} |
| Tracking | IDF1 | {mot['idf1']:.3f} |
| Tracking | ID switches | {mot['id_switches']} |
| Possession | frame agreement vs GT | {100 * agree:.1f}% |
| Events | pass precision | {pass_prec:.2f} |
| Events | pass recall | {pass_rec:.2f} |
"""
    for team, gt_scheme in (("home", "zonal/press"), ("away", "man")):
        op = marking.get(team, {}).get("open_play", {})
        md += (f"| Marking | {team} man-score (GT: {gt_scheme}) | "
               f"{op.get('team_man_score', 'n/a')} ({op.get('scheme', '?')}) |\n")

    md += """
Notes:
- Rendered synthetic broadcast: real pipelines (line/conic calibration,
  ByteTrack, colour team clustering) on ideal imagery — an upper bound.
- MOTA/IDF1 use fixed-size foot-point proxy boxes (IoU 0.2) because the
  tracking table stores projected foot points, not raw boxes.
- Real-broadcast performance is qualitatively validated; see README
  limitations for the honest gap list.
"""
    out_md = REPO / "docs" / "validation.md"
    out_md.parent.mkdir(exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"written to {out_md}")


if __name__ == "__main__":
    main()
