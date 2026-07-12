"""Process a video clip end-to-end into a job directory (CLI equivalent of the
app's upload path).

Runs the full pipeline — perception → analytics → intelligence → report →
media — writing every artifact through the ArtifactStore so the result is
immediately browsable in the Streamlit app (it lists any directory under
``data/jobs`` or ``data/demo``).

Examples
--------
    # bundled fallback stack (COCO/blob, line calibration)
    python scripts/process_clip.py data/raw/match.mp4

    # the trained product stack on GPU
    python scripts/process_clip.py data/raw/rma_mc_50s.mp4 \
        --name rma-mc-full \
        --detector weights/football_yolo11n.pt \
        --keypoints weights/pitch_keypoints.pt \
        --device cuda --jersey
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to the input clip")
    ap.add_argument("--name", default=None,
                    help="job dir name under data/jobs (default: clip stem)")
    ap.add_argument("--detector", default=None,
                    help="fine-tuned YOLO weights (default: auto/COCO fallback)")
    ap.add_argument("--keypoints", default=None,
                    help="pitch-keypoint weights (default: line/conic calibration)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--jersey", action="store_true", help="enable jersey OCR")
    ap.add_argument("--config", default=None,
                    help="config yaml; defaults to configs/football.yaml when "
                         "present (the product config), else built-in defaults")
    ap.add_argument("--ball-tracknet", default=None,
                    help="TrackNet ball weights (.pt); omit for YOLO+Kalman ball")
    ap.add_argument("--teams", choices=["kmeans_lab", "embed"], default=None,
                    help="team assignment method (embed = learned crop embeddings)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--target-fps", type=float, default=25.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("process_clip")

    from pitchiq.config import load_config
    from pitchiq.core.artifacts import ArtifactStore
    from pitchiq.pipeline.full import FullPipeline

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    name = args.name or video.stem
    job_dir = REPO / "data" / "jobs" / name
    store = ArtifactStore(job_dir)

    detection: dict = {"imgsz": args.imgsz, "device": args.device}
    if args.detector:
        detection.update(backend="yolo", weights=args.detector)
    if args.ball_tracknet:
        detection["ball"] = {"tracknet_weights": args.ball_tracknet}
    overrides: dict = {
        "detection": detection,
        "jersey": {"enabled": bool(args.jersey)},
        "video": {"target_fps": args.target_fps, "max_frames": args.max_frames},
    }
    if args.keypoints:
        overrides["calibration"] = {"keypoint_weights": args.keypoints}
    if args.teams:
        overrides["teams"] = {"method": args.teams}
    cfg_path = args.config
    if cfg_path is None:
        product = REPO / "configs" / "football.yaml"
        cfg_path = product if product.exists() else None
        if cfg_path:
            log.info("using product config %s (override with --config)", cfg_path)
    cfg = load_config(path=cfg_path, overrides=overrides)

    log.info("processing %s -> %s (detector=%s, keypoints=%s, device=%s)",
             video.name, job_dir, args.detector or "auto", args.keypoints or "lines",
             args.device)
    t0 = time.time()
    FullPipeline(cfg).process_video(video, store)
    log.info("done in %.0fs — open the app and select '%s'", time.time() - t0, name)

    # quick sanity summary
    df = store.load_tracking()
    meta = store.load_meta()
    print("\n=== summary ===")
    print("frames:", df.frame.nunique(), "| rows:", len(df))
    print("classes:", df["class"].value_counts().to_dict())
    print("teams:", df[df["class"] == "player"]["team"].value_counts().to_dict())
    print("kit colours:", meta.kit_colors, "| separability:",
          round(meta.extras.get("team_separability", float("nan")), 2))
    hdf = store.load_homography()
    print("calibration methods:", hdf["method"].value_counts().to_dict())
    if meta.extras.get("notes"):
        print("notes:", meta.extras["notes"])


if __name__ == "__main__":
    main()
