"""Build the bundled demo match(es) end-to-end.

Two variants of the same simulated fixture:

* ``synthetic-derby`` — ground-truth tracking straight from the simulator
  (positions exact), rendered broadcast video, full analytics → intelligence
  → report → media. Fast (~5 min) and deterministic; this is the match the
  app ships with.
* ``synthetic-derby-cv`` (``--cv``) — the SAME rendered video pushed through
  the real perception stack (detection → tracking → calibration), so the
  dashboard shows genuine raw-video→dashboard output and the two variants can
  be compared side by side. CPU-heavy (~30-45 min).

Also trains the contrastive style encoder on simulated matches if weights are
missing, so the intelligence tab demonstrates the learned (6.1b) embedding.

Usage:
    python scripts/build_demo.py [--cv] [--skip-encoder] [--half-minutes 2.0]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_demo")

ENCODER_WEIGHTS = REPO / "weights" / "style_encoder.pt"


def fill_pixel_coords(tracking: pd.DataFrame, homographies: list[dict]) -> pd.DataFrame:
    """Project GT pitch coords into pixels via the renderer's exact H (per frame)."""
    from pitchiq.core.geometry import apply_homography

    H_by_frame = {r["frame"]: np.linalg.inv(r["H"]) for r in homographies}
    df = tracking.copy()
    out_x = np.full(len(df), np.nan, dtype=np.float32)
    out_y = np.full(len(df), np.nan, dtype=np.float32)
    for f, idx in df.groupby("frame").indices.items():
        Hinv = H_by_frame.get(int(f))
        if Hinv is None:
            continue
        pts = df.iloc[idx][["x_pitch", "y_pitch"]].to_numpy(dtype=float)
        px = apply_homography(Hinv, pts)
        ok = np.isfinite(px).all(axis=1)
        # only keep pixels inside a sane frame envelope
        ok &= (px[:, 0] > -200) & (px[:, 0] < 1300) & (px[:, 1] > -100) & (px[:, 1] < 700)
        out_x[idx[ok]] = px[ok, 0]
        out_y[idx[ok]] = px[ok, 1]
    df["x_pixel"] = out_x
    df["y_pixel"] = out_y
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true", help="also build the CV-pipeline variant")
    ap.add_argument("--skip-encoder", action="store_true")
    ap.add_argument("--half-minutes", type=float, default=2.0)
    ap.add_argument("--fresh", action="store_true", help="delete existing demo dirs first")
    args = ap.parse_args()

    from pitchiq.config import load_config
    from pitchiq.core.artifacts import ArtifactStore
    from pitchiq.core.pitch import Pitch
    from pitchiq.core.schema import homographies_to_frame
    from pitchiq.demo.render import render_match
    from pitchiq.demo.simulate import simulate_demo_match
    from pitchiq.pipeline.full import FullPipeline

    # 1. learned style encoder (licence-clean: trained on simulations)
    if not args.skip_encoder and not ENCODER_WEIGHTS.exists():
        log.info("training style encoder (one-time, ~2-4 min CPU)...")
        subprocess.run([sys.executable, str(REPO / "scripts" / "train_style_encoder.py"),
                        "--out", str(ENCODER_WEIGHTS)], check=True)

    overrides: dict = {"simulator": {"half_minutes": args.half_minutes}}
    if ENCODER_WEIGHTS.exists():
        overrides["embeddings"] = {"learned": {"enabled": True,
                                               "weights": str(ENCODER_WEIGHTS)}}
    cfg = load_config(overrides=overrides)

    demo_root = REPO / cfg.app.demo_root
    gt_dir = demo_root / "synthetic-derby"
    if args.fresh and gt_dir.exists():
        shutil.rmtree(gt_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(gt_dir)

    # 2. simulate + render
    log.info("simulating fixture (%.1f min halves)...", args.half_minutes)
    sim = simulate_demo_match(cfg.simulator)
    log.info("rendering broadcast view...")
    rr = render_match(sim.tracking, Pitch(), sim.meta.kit_colors,
                      store.input_video, cfg.simulator.fps)

    # 3. ground-truth store: tracking (with pixels), GT homographies, meta, GT extras
    tracking = fill_pixel_coords(sim.tracking, rr.homographies)
    store.save_tracking(tracking)
    store.save_homography(homographies_to_frame(rr.homographies))
    sim.meta.extras["marking_gt"] = sim.marking_gt
    sim.meta.extras["variant"] = "ground-truth"
    store.save_meta(sim.meta)
    sim.events.to_parquet(gt_dir / "events_gt.parquet", index=False)

    # 4. analytics → intelligence → report → media
    log.info("running analytics/intelligence/report/media on GT variant...")
    FullPipeline(cfg).process_tracking_only(store, video_path=store.input_video)
    log.info("GT demo ready at %s", gt_dir)

    # 5. optional CV variant: real perception on the rendered video
    if args.cv:
        cv_dir = demo_root / "synthetic-derby-cv"
        if args.fresh and cv_dir.exists():
            shutil.rmtree(cv_dir)
        cv_dir.mkdir(parents=True, exist_ok=True)
        cv_store = ArtifactStore(cv_dir)
        shutil.copy(store.input_video, cv_store.input_video)
        cv_cfg = load_config(overrides={**overrides,
                                        "detection": {"backend": "blob"},
                                        "jersey": {"enabled": False}})
        log.info("running FULL CV pipeline on rendered video (long)...")
        FullPipeline(cv_cfg).process_video(cv_store.input_video, cv_store)
        # keep the repo lean: the CV variant is derivable, so tag its meta
        meta = cv_store.load_meta()
        meta.team_names = {"home": "Crimson City", "away": "Azure United"}
        meta.extras["variant"] = "cv-pipeline"
        cv_store.save_meta(meta)
        log.info("CV demo ready at %s", cv_dir)


if __name__ == "__main__":
    main()
