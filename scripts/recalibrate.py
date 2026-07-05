"""Re-run ONLY calibration on a processed match, reprojecting cached pixels.

The expensive perception work (detection + tracking) is already cached in the
tracking table's pixel columns; calibration quality is the thing that
improves over time (e.g. after training the keypoint model). This tool
re-estimates per-frame homographies from the video, reprojects every cached
pixel to pitch coordinates, and re-runs the downstream analytics — minutes
instead of an hour.

    python scripts/recalibrate.py data/demo/synthetic-derby-cv \
        [--keypoint-weights weights/pitch_keypoints.pt] [--every-n 1]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--keypoint-weights", default=None)
    ap.add_argument("--every-n", type=int, default=None)
    ap.add_argument("--no-downstream", action="store_true",
                    help="only recalibrate + reproject; skip analytics rerun")
    args = ap.parse_args()

    from pitchiq.config import load_config
    from pitchiq.core.artifacts import ArtifactStore
    from pitchiq.core.geometry import apply_homography
    from pitchiq.core.pitch import Pitch
    from pitchiq.core.schema import homographies_to_frame, homography_for_frame
    from pitchiq.core.video import FrameReader
    from pitchiq.perception.calibration import PitchCalibrator
    from pitchiq.perception.tracking.camera_motion import CameraMotionEstimator

    store = ArtifactStore(args.job_dir)
    assert store.has_tracking() and store.input_video.exists(), \
        "needs cached tracking + input video"
    overrides: dict = {}
    if args.keypoint_weights:
        overrides["calibration"] = {"keypoint_weights": args.keypoint_weights,
                                    "method": "auto"}
    if args.every_n:
        overrides.setdefault("calibration", {})["every_n_frames"] = args.every_n
    cfg = load_config(overrides=overrides)

    meta = store.load_meta()
    pitch = Pitch(meta.pitch_length, meta.pitch_width)
    calibrator = PitchCalibrator(cfg.calibration, pitch)
    cm = CameraMotionEstimator()

    df = store.load_tracking()
    boxes_by_frame = {f: g[["x_pixel", "y_pixel"]].to_numpy()
                      for f, g in df.dropna(subset=["x_pixel"]).groupby("frame")}

    print("re-estimating homographies...")
    t0 = time.time()
    hrecs = []
    reader = FrameReader(store.input_video, target_fps=meta.fps)
    for idx, ts, frame in reader:
        A = cm.estimate(frame, [])
        res = calibrator.process(idx, frame, camera_affine=A)
        if res.is_scene_cut:
            cm.reset()
        hrecs.append(dict(frame=idx, timestamp=ts, H=res.H,
                          reproj_error_px=res.reproj_error_px,
                          method=res.method, is_scene_cut=res.is_scene_cut))
        if idx % 500 == 0:
            n_direct = sum(1 for r in hrecs if r["method"] not in ("flow", "none"))
            print(f"  frame {idx} ({(idx + 1) / (time.time() - t0):.1f} fps, "
                  f"{n_direct} direct estimates)")
    hdf = homographies_to_frame(hrecs)
    store.save_homography(hdf)

    print("reprojecting cached pixels...")
    H_cache: dict[int, np.ndarray | None] = {}
    new_x = df["x_pitch"].to_numpy().copy()
    new_y = df["y_pitch"].to_numpy().copy()
    frames_arr = df["frame"].to_numpy()
    px = df["x_pixel"].to_numpy()
    py = df["y_pixel"].to_numpy()
    for f in np.unique(frames_arr):
        H_cache[int(f)] = homography_for_frame(hdf, int(f))
    for i in range(len(df)):
        H = H_cache.get(int(frames_arr[i]))
        if H is None or not np.isfinite(px[i]):
            continue
        pt = apply_homography(H, [[px[i], py[i]]])[0]
        if np.all(np.isfinite(pt)) and pitch.contains(pt[0], pt[1], margin=6.0):
            new_x[i], new_y[i] = pt
        else:
            new_x[i], new_y[i] = np.nan, np.nan
    df["x_pitch"] = new_x.astype("float32")
    df["y_pitch"] = new_y.astype("float32")
    store.save_tracking(df)

    methods = hdf["method"].value_counts().to_dict()
    print(f"done in {time.time() - t0:.0f}s — methods: {methods}")

    if not args.no_downstream:
        from pitchiq.pipeline.full import FullPipeline

        print("re-running analytics → report on reprojected table...")
        FullPipeline(cfg).process_tracking_only(store, video_path=store.input_video)
    print("recalibration complete.")


if __name__ == "__main__":
    main()
