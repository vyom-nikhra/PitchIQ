"""Layer-1 orchestration: video → tracking table (+ homographies + meta).

One pass over the frames runs detection → appearance features → camera-motion
estimation → calibration → tracking → ball selection, sampling kit colours
and jersey crops along the way. Team/jersey decisions are *global* (they need
the whole clip), so they are resolved after the pass and joined back onto the
per-frame rows. Everything is persisted through the
:class:`~pitchiq.core.artifacts.ArtifactStore` so Layers 2-3 never re-run CV.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pitchiq.config import Config
from pitchiq.core.artifacts import ArtifactStore
from pitchiq.core.geometry import apply_homography
from pitchiq.core.pitch import Pitch
from pitchiq.core.schema import (
    BALL_ID,
    MatchMeta,
    homographies_to_frame,
    validate_tracking_table,
)
from pitchiq.core.types import EntityClass, Team
from pitchiq.core.video import FrameReader
from pitchiq.perception.calibration import PitchCalibrator
from pitchiq.perception.detection import create_detector
from pitchiq.perception.detection.ball import (
    BallSelector,
    interpolate_ball,
    refine_ball_track,
)
from pitchiq.perception.jersey import JerseyVoter, create_jersey_reader
from pitchiq.perception.teams import TeamAssigner
from pitchiq.perception.teams.assign import torso_crop
from pitchiq.perception.tracking import ByteTracker
from pitchiq.perception.tracking.appearance import create_embedder
from pitchiq.perception.tracking.camera_motion import CameraMotionEstimator

log = logging.getLogger(__name__)

TEAM_SAMPLE_EVERY = 5   # frames between kit-colour samples per pass
JERSEY_SAMPLE_EVERY = 10


class PerceptionPipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.pitch = Pitch(cfg.pitch.length_m, cfg.pitch.width_m)
        self.detector = create_detector(cfg.detection)
        self.embedder = create_embedder(cfg.tracking.appearance)
        self.tracker = ByteTracker(cfg.tracking)
        self.ball = BallSelector(cfg.detection.ball, detector=self.detector)
        # optional TrackNet heatmap ball tracker (falls back to BallSelector)
        self.tracknet = None
        if cfg.detection.ball.tracknet_weights:
            try:
                from pitchiq.perception.detection.tracknet import TrackNetBall

                self.tracknet = TrackNetBall(
                    cfg.detection.ball.tracknet_weights,
                    device=cfg.detection.device,
                    peak_threshold=cfg.detection.ball.tracknet_threshold)
            except Exception as exc:
                log.warning("TrackNet unavailable (%s); using YOLO+Kalman ball", exc)
        self.calibrator = PitchCalibrator(cfg.calibration, self.pitch)
        self.camera_motion = (
            CameraMotionEstimator() if cfg.tracking.camera_motion_compensation else None
        )
        self.teams = TeamAssigner(cfg.teams)
        self.jersey_reader = create_jersey_reader(cfg.jersey)
        self.jersey_votes = JerseyVoter(cfg.jersey)
        self.pose = None
        if cfg.pose.enabled:
            from pitchiq.perception.pose import PoseSampler

            sampler = PoseSampler(cfg.pose.model, cfg.detection.device,
                                  cfg.pose.min_kp_conf)
            self.pose = sampler if sampler.ok else None

    # ----------------------------------------------------------------- run
    def run(
        self,
        video_path: str | Path,
        store: ArtifactStore,
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, MatchMeta]:
        reader = FrameReader(
            video_path, self.cfg.video.target_fps, self.cfg.video.max_frames
        )
        n_est = max(reader.n_frames_estimate, 1)
        frame_dt = 1.0 / max(reader.fps, 1e-6)
        rows: list[dict] = []
        hrecs: list[dict] = []
        cls_votes: dict[int, dict[EntityClass, float]] = defaultdict(lambda: defaultdict(float))
        t0 = time.time()

        for idx, ts, frame in reader:
            dets = self.detector.detect(frame)
            person_dets = [d for d in dets if d.cls != EntityClass.BALL]
            if self.embedder is not None:
                for d in person_dets:
                    d.feature = self.embedder.embed(frame, d.bbox)

            A = None
            if self.camera_motion is not None:
                A = self.camera_motion.estimate(frame, [d.bbox for d in person_dets])

            calib = self.calibrator.process(idx, frame, camera_affine=A)
            if calib.is_scene_cut and self.camera_motion is not None:
                self.camera_motion.reset()
                A = None

            tracks = self.tracker.update(
                person_dets, camera_affine=None if calib.is_scene_cut else A,
                homography=calib.H, dt=frame_dt, scene_cut=calib.is_scene_cut)

            # ball: prefer the TrackNet heatmap tracker when configured, else the
            # YOLO+Kalman selector. Both yield (x_px, y_px, ground_y_px, conf).
            ball_xy = None
            if self.tracknet is not None:
                if calib.is_scene_cut:
                    self.tracknet.reset()
                bt = self.tracknet.detect(frame)
                if bt is not None:
                    ball_xy = (bt[0], bt[1], bt[1], bt[2])  # ball centre on ground plane
            else:
                ball_det = self.ball.select(frame, dets)
                if ball_det is not None:
                    bx, by = ball_det.center
                    ball_xy = (bx, by, float(ball_det.bbox[3]), float(ball_det.conf))

            H = calib.H
            for t in tracks:
                fx, fy = t.bbox[[0, 2]].mean(), t.bbox[3]
                px, py = (np.nan, np.nan)
                if H is not None:
                    proj = apply_homography(H, [[fx, fy]])[0]
                    if np.all(np.isfinite(proj)) and self.pitch.contains(proj[0], proj[1], margin=6.0):
                        px, py = float(proj[0]), float(proj[1])
                rows.append(
                    dict(frame=idx, timestamp=ts, entity_id=t.track_id, cls=t.entity_class,
                         x_pixel=float(fx), y_pixel=float(fy), x_pitch=px, y_pitch=py,
                         conf=float(t.score))
                )
                cls_votes[t.track_id][t.entity_class] += t.score
                if idx % TEAM_SAMPLE_EVERY == 0:
                    self.teams.add_sample(t.track_id, frame, t.bbox)
                if (
                    self.jersey_reader is not None
                    and idx % JERSEY_SAMPLE_EVERY == 0
                    and (t.bbox[3] - t.bbox[1]) >= self.cfg.jersey.min_height_px
                ):
                    crop = torso_crop(frame, t.bbox, self.cfg.teams)
                    if crop is not None:
                        read = self.jersey_reader.read(crop)
                        if read is not None:
                            self.jersey_votes.add(t.track_id, read[0], read[1])

            if ball_xy is not None:
                bx, by, ground_y, bconf = ball_xy
                px, py = (np.nan, np.nan)
                if H is not None:
                    proj = apply_homography(H, [[bx, ground_y]])[0]
                    if np.all(np.isfinite(proj)) and self.pitch.contains(proj[0], proj[1], margin=6.0):
                        px, py = float(proj[0]), float(proj[1])
                rows.append(
                    dict(frame=idx, timestamp=ts, entity_id=BALL_ID, cls=EntityClass.BALL,
                         x_pixel=float(bx), y_pixel=float(by), x_pitch=px, y_pitch=py,
                         conf=float(bconf))
                )

            if self.pose is not None and idx % self.cfg.pose.sample_every == 0:
                self.pose.sample(frame, tracks)

            hrecs.append(
                dict(frame=idx, timestamp=ts, H=H, reproj_error_px=calib.reproj_error_px,
                     method=calib.method, is_scene_cut=calib.is_scene_cut)
            )
            if progress_cb and (idx % 20 == 0 or idx == n_est - 1):
                fps_proc = (idx + 1) / max(time.time() - t0, 1e-6)
                progress_cb(min(0.98, idx / n_est), f"frame {idx + 1}/{n_est} ({fps_proc:.1f} fps)")

        if not rows:
            raise RuntimeError("perception produced no rows — is the video readable?")
        cut_frames = {int(r["frame"]) for r in hrecs if r["is_scene_cut"]}
        df, meta = self._finalize(rows, cls_votes, reader, cut_frames)
        hdf = homographies_to_frame(hrecs)

        store.save_tracking(df)
        store.save_homography(hdf)
        store.save_meta(meta)
        if self.pose is not None:
            pose_df = self.pose.finalize()
            if len(pose_df):
                pose_df.to_parquet(store.pose_path, index=False)
                log.info("pose descriptors for %d tracks", len(pose_df))
        if progress_cb:
            progress_cb(1.0, "perception complete")
        return df, hdf, meta

    # ------------------------------------------------------------ finalize
    def _finalize(self, rows, cls_votes, reader,
                  cut_frames: set[int] | None = None) -> tuple[pd.DataFrame, MatchMeta]:
        raw = pd.DataFrame(rows)

        # drop fragment tracks
        counts = raw.loc[raw["entity_id"] != BALL_ID, "entity_id"].value_counts()
        keep = set(counts[counts >= self.cfg.tracking.min_track_len].index) | {BALL_ID}
        raw = raw[raw["entity_id"].isin(keep)].copy()

        track_cls = {tid: max(v, key=v.get) for tid, v in cls_votes.items() if tid in keep}
        mean_x = (
            raw[raw["entity_id"] != BALL_ID]
            .groupby("entity_id")["x_pitch"]
            .mean()
            .to_dict()
        )
        assignment = self.teams.finalize(track_cls, mean_x, self.pitch.length)
        for tid, cls in assignment.cls_override.items():
            track_cls[tid] = cls
        numbers = self.jersey_votes.finalize()

        raw["cls"] = raw.apply(
            lambda r: EntityClass.BALL if r["entity_id"] == BALL_ID else track_cls.get(r["entity_id"], r["cls"]),
            axis=1,
        )
        raw["team"] = raw["entity_id"].map(
            lambda tid: Team.NONE if tid == BALL_ID else assignment.team_of_track.get(tid, Team.NONE)
        )
        raw["jersey_no"] = raw["entity_id"].map(lambda tid: numbers.get(tid))

        df = pd.DataFrame(
            {
                "frame": raw["frame"],
                "timestamp": raw["timestamp"],
                "entity_id": raw["entity_id"],
                "class": raw["cls"].map(lambda c: c.value if isinstance(c, EntityClass) else str(c)),
                "team": raw["team"].map(lambda t: t.value if isinstance(t, Team) else str(t)),
                "jersey_no": raw["jersey_no"],
                "x_pixel": raw["x_pixel"],
                "y_pixel": raw["y_pixel"],
                "x_pitch": raw["x_pitch"],
                "y_pitch": raw["y_pitch"],
                "conf": raw["conf"],
            }
        )
        df = validate_tracking_table(df)
        if self.cfg.detection.ball.postprocess:
            # outliers first, so interpolation never bridges *toward* one
            df = refine_ball_track(df, BALL_ID, cut_frames)
        # exact observed-ball count (pre-interpolation) for the quality report
        ball_observed = int(df.loc[df["entity_id"] == BALL_ID, "frame"].nunique())
        df = interpolate_ball(df, self.cfg.detection.ball.max_gap_interpolate, BALL_ID)
        df = validate_tracking_table(df)

        # attack directions: the team defending the low-x side attacks +x
        tm = df[(df["team"] == "home") | (df["team"] == "away")]
        home_x = tm.loc[tm["team"] == "home", "x_pitch"].mean()
        away_x = tm.loc[tm["team"] == "away", "x_pitch"].mean()
        home_attacks_pos = bool(np.nan_to_num(home_x, nan=0.0) <= np.nan_to_num(away_x, nan=0.0))
        meta = MatchMeta(
            fps=reader.fps,
            n_frames=int(df["frame"].max()) + 1,
            pitch_length=self.pitch.length,
            pitch_width=self.pitch.width,
            kit_colors={**{"home": "#d62728", "away": "#1f77b4"}, **assignment.kit_colors},
            attack_direction={"home": 1 if home_attacks_pos else -1,
                              "away": -1 if home_attacks_pos else 1},
            source=f"video:{Path(reader.path).name}|detector:{self.detector.name}",
            extras={
                "team_separability": assignment.separability,
                "notes": assignment.notes,
                "ball_observed_frames": ball_observed,
            },
        )
        return df, meta
