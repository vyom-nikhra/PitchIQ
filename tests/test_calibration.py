"""Calibration integration test on a rendered synthetic view with known H."""

import numpy as np
import pandas as pd
import pytest

from pitchiq.core.geometry import apply_homography
from pitchiq.demo.render import BroadcastRenderer
from pitchiq.perception.calibration.estimate import LineCalibrator


@pytest.fixture(scope="module")
def rendered_view(pitch):
    """One synthetic broadcast frame (no players) + its exact GT homography."""
    r = BroadcastRenderer(pitch, {"home": "#d62728", "away": "#1f77b4"}, 1024, 576)
    r.cam.update(np.array([52.5, 34.0]))
    empty = pd.DataFrame(columns=["entity_id", "class", "team", "x_pitch", "y_pitch", "jersey_no"])
    img, _ = r.render_frame(empty, None)
    return img, r.cam.ground_homography()


def test_line_calibrator_recovers_homography(pitch, rendered_view):
    img, H_gt = rendered_view
    hyp = LineCalibrator(pitch).estimate(img)
    assert hyp is not None, "calibration failed on a clean centre view"
    test_w = np.array([[35, 20], [52.5, 34], [70, 50], [45, 55]], dtype=float)
    img_pts = apply_homography(np.linalg.inv(H_gt), test_w)
    proj = apply_homography(hyp.H, img_pts)
    err = np.linalg.norm(proj - test_w, axis=1)
    assert np.nanmean(err) < 1.0, f"mean error {np.nanmean(err):.2f} m"


def test_degenerate_homography_rejected(pitch, rendered_view):
    """A collapsed H must not pass the plausibility gates regardless of mask."""
    img, _ = rendered_view
    cal = LineCalibrator(pitch)
    cal.estimate(img)  # populate mask caches
    H_collapse = np.array([[1e-4, 0, 16.5], [0, 1e-4, 14.0], [0, 0, 1.0]])
    assert not cal._plausible_geometry(H_collapse, 1024, 576)
    assert cal.score_homography(H_collapse) <= 0.0


def test_smoother_mirror_canonicalisation(pitch, rendered_view):
    from pitchiq.perception.calibration.temporal import HomographySmoother, _symmetry_transforms

    _, H_gt = rendered_view
    sm = HomographySmoother(pitch)
    sm.update(H_gt, hard=True)
    mirrored = _symmetry_transforms(pitch)[3] @ H_gt  # 180° twin
    fixed = sm.canonicalize(mirrored)
    test_w = np.array([[30.0, 20.0]])
    img_pt = apply_homography(np.linalg.inv(H_gt), test_w)
    assert np.allclose(apply_homography(fixed, img_pt), test_w, atol=1e-6)


def test_scene_cut_detector():
    from pitchiq.perception.calibration.temporal import SceneCutDetector

    det = SceneCutDetector(threshold=0.45)
    rng = np.random.default_rng(0)
    # textured pitch-like frame (uniform frames have degenerate histograms)
    base = np.full((90, 160, 3), (40, 130, 50), dtype=np.int16)
    pitch_view = np.clip(base + rng.integers(-25, 25, base.shape), 0, 255).astype(np.uint8)
    panned = np.roll(pitch_view, 12, axis=1)  # same scene, camera pan
    crowd = rng.integers(0, 255, (90, 160, 3)).astype(np.uint8)
    assert det(pitch_view) is False     # first frame: never a cut
    assert det(panned) is False         # same scene
    assert det(crowd) is True           # hard cut


def test_keypoint_calibrator_init_error_contract(pitch, tmp_path):
    """Missing weights raise FileNotFoundError (PitchCalibrator's graceful-
    fallback trigger); a corrupt file raises something else — a genuine error
    that must not silently downgrade to line-based calibration."""
    pytest.importorskip("torch")
    from pitchiq.perception.calibration.calibrator import PitchCalibrator
    from pitchiq.perception.calibration.keypoints import KeypointCalibrator
    from pitchiq.config import CalibrationConfig

    with pytest.raises(FileNotFoundError):
        KeypointCalibrator(pitch, str(tmp_path / "nope.pt"))

    # absent weights: PitchCalibrator degrades to line-based, documented
    cfg = CalibrationConfig(keypoint_weights=str(tmp_path / "nope.pt"))
    assert PitchCalibrator(cfg, pitch).keypoints is None

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"this is not a checkpoint")
    with pytest.raises(Exception) as exc_info:
        PitchCalibrator(CalibrationConfig(keypoint_weights=str(corrupt)), pitch)
    assert not isinstance(exc_info.value, (FileNotFoundError, ImportError))
