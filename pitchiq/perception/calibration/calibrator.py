"""Per-frame calibration orchestrator.

Strategy per frame (config: ``calibration``):

1. Scene-cut check — a cut resets smoothing/propagation state.
2. Every ``every_n_frames`` (or when there is no valid solution) run a *full
   estimation*: the learned keypoint model when weights are configured,
   otherwise the line-based estimator (with the propagated solution as a
   hint to skip the combinatorial search).
3. Between estimations — or when estimation fails — the last solution is
   propagated by the frame-to-frame camera affine.
4. Fresh estimates are blended by the point-space smoother; quality gates
   (reprojection error, mask score) reject bad estimates so a wrong
   calibration never silently replaces a good propagated one.

Every frame gets an auditable :class:`CalibrationResult` with the method that
produced it (``keypoints`` | ``lines`` | ``flow`` | ``none``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from pitchiq.config import CalibrationConfig
from pitchiq.core.pitch import Pitch
from pitchiq.perception.calibration.estimate import LineCalibrator
from pitchiq.perception.calibration.temporal import HomographySmoother, SceneCutDetector

log = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    H: np.ndarray | None
    reproj_error_px: float
    method: str  # keypoints | lines | flow | none
    is_scene_cut: bool
    score: float = 0.0


class PitchCalibrator:
    def __init__(self, cfg: CalibrationConfig, pitch: Pitch) -> None:
        self.cfg = cfg
        self.pitch = pitch
        self.cut_detector = SceneCutDetector(cfg.scene_cut_threshold)
        self.smoother = HomographySmoother(pitch, cfg.smoothing.alpha)
        self.lines = LineCalibrator(pitch, min_line_score=cfg.min_line_score)
        self.keypoints = None
        if cfg.method in ("keypoints", "auto") and cfg.keypoint_weights:
            try:
                from pitchiq.perception.calibration.keypoints import KeypointCalibrator

                self.keypoints = KeypointCalibrator(pitch, cfg.keypoint_weights)
            except Exception as exc:
                log.warning("keypoint calibrator unavailable (%s); using line-based", exc)
        self._frames_since_estimate = 10**9
        self._last_error = float("nan")

    def reset(self) -> None:
        self.smoother.reset()
        self._frames_since_estimate = 10**9
        self._last_error = float("nan")

    def process(
        self, frame_idx: int, frame_bgr: np.ndarray, camera_affine: np.ndarray | None = None
    ) -> CalibrationResult:
        is_cut = self.cut_detector(frame_bgr)
        if is_cut:
            self.reset()

        # 1. keep the incumbent solution current under camera motion
        if self.smoother.current is not None and self.cfg.propagate_with_flow:
            if camera_affine is not None:
                self.smoother.propagate(camera_affine)
        have_solution = self.smoother.current is not None
        due = self._frames_since_estimate >= self.cfg.every_n_frames
        self._frames_since_estimate += 1

        # 2. periodically try to (re-)anchor with a fresh full estimate
        if due or not have_solution:
            est = self._full_estimate(frame_bgr)
            if est is not None and est[1] <= self.cfg.max_reproj_error_px:
                H_est, err, method, score = est
                incumbent = self.smoother.current
                # judge the incumbent on the SAME frame's evidence
                s_old = self.lines.score_homography(incumbent) if incumbent is not None else -1.0
                if incumbent is None:
                    H = self.smoother.update(H_est, hard=True)
                elif score >= max(1.15 * s_old, 0.0):
                    # decisively better evidence -> snap; comparable -> blend
                    hard = s_old < 0.12 or score > 2.0 * max(s_old, 1e-6)
                    H = self.smoother.update(H_est, hard=hard)
                else:
                    H = incumbent  # fresh estimate is no better than propagation
                    method, score = "flow", s_old
                self._frames_since_estimate = 0
                self._last_error = err
                return CalibrationResult(H, err, method, is_cut, score)
            # estimation failed → fall through to propagation

        # 3. propagation-only frame
        if have_solution:
            H = self.smoother.current
            if H is not None:
                return CalibrationResult(H, self._last_error, "flow", is_cut)
        return CalibrationResult(None, float("nan"), "none", is_cut)

    def _full_estimate(self, frame_bgr: np.ndarray):
        """Returns (H, reproj_error_px, method, score) or None.

        Sparse-evidence hypotheses (few matched lines, no conic) must clear a
        higher score floor: with only ~4 intersections a wrong template
        assignment can still cover the few visible mask lines, and a single
        accepted mis-calibration would poison the smoothed solution.
        """
        if self.keypoints is not None:
            got = self.keypoints.estimate(frame_bgr)
            if got is not None:
                H, err, _ = got
                return H, err, "keypoints", 1.0
        hint = self.smoother.current
        hyp = self.lines.estimate(frame_bgr, hint_H=hint)
        if hyp is None:
            return None
        strong_evidence = hyp.kind in ("circle", "hint") or hyp.n_lines >= 4
        floor = self.cfg.min_line_score if strong_evidence else max(0.45, 1.6 * self.cfg.min_line_score)
        gate_score = hyp.raw_score if hyp.raw_score >= 0 else hyp.score
        if gate_score < floor:
            return None
        return hyp.H, hyp.reproj_error_px, "lines", hyp.score
