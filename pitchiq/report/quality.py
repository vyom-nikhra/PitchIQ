"""Perception-quality assessment: how much should you trust this tracking table?

Computed purely from cached artifacts (tracking table + homography table +
meta) so it works on any job — including pre-baked demos — without re-running
CV. The output feeds three consumers:

* ``facts["data_quality"]`` — the LLM report hedges ball-dependent claims
  when quality is low instead of laundering perception noise into confident
  prose;
* the dashboard header badge ("Tracking confidence: high/medium/low");
* Q&A retrieval (via the flattened facts corpus).

Levels are deliberately coarse (high / medium / low) with the underlying
numbers always attached: the levels drive tone, the numbers stay auditable.
Thresholds were calibrated against four reference jobs: the ground-truth and
blob-CV synthetic demos, and a real SoccerNet clip run with the trained and
fallback ball stacks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitchiq.core.artifacts import ArtifactStore
from pitchiq.core.schema import BALL_ID, MatchMeta

LEVELS = ("low", "medium", "high")
_RANK = {name: i for i, name in enumerate(LEVELS)}

# Calibration: direct-solve reprojection error (px) and share of frames with
# any homography. Flow-bridged frames count toward coverage but not error —
# their stored residual is propagated, not measured.
CALIB_COVERAGE = {"high": 0.90, "medium": 0.60}
CALIB_REPROJ_PX = {"high": 25.0, "medium": 60.0}

# Ball: share of frames where the ball was actually *observed* (not bridged
# by interpolation). Exact for new runs (meta.extras["ball_observed_frames"]);
# estimated from the interpolation confidence decay for older jobs.
BALL_OBSERVED = {"high": 0.70, "medium": 0.45}

# Tracking: median track length in seconds. Short-lived tracks mean identity
# churn; on partial-view broadcast some churn is inherent (players leave
# frame), so this is a coarse gate, not a MOT score.
TRACK_LEN_S = {"high": 5.0, "medium": 2.0}

# Teams: kit-separability score written by the team assigner.
TEAM_SEP = {"high": 3.0, "medium": 1.5}


def _grade(value: float | None, thresholds: dict[str, float],
           higher_is_better: bool = True) -> str:
    if value is None or not np.isfinite(value):
        return "low"
    if higher_is_better:
        if value >= thresholds["high"]:
            return "high"
        return "medium" if value >= thresholds["medium"] else "low"
    if value <= thresholds["high"]:
        return "high"
    return "medium" if value <= thresholds["medium"] else "low"


def _worst(*levels: str) -> str:
    return LEVELS[min(_RANK[lv] for lv in levels)]


def assess_quality(tracking: pd.DataFrame, homography: pd.DataFrame | None,
                   meta: MatchMeta) -> dict:
    """Score perception quality from cached artifacts. Never raises on
    missing optional signals — every absence is graded and named instead."""
    is_cv = str(meta.source or "").startswith("video:")
    detector = ""
    if "detector:" in (meta.source or ""):
        detector = meta.source.split("detector:", 1)[1]

    if not is_cv:
        # Simulator ground truth or a data-provider import: positions are
        # exact by construction, so perception quality is not the caveat.
        return {
            "overall": "high",
            "is_cv": False,
            "levels": {"calibration": "high", "ball": "high",
                       "tracking": "high", "teams": "high"},
            "components": {"source": meta.source},
            "notes": ["Positions come from ground truth (simulator or data "
                      "provider), not from computer vision — treat the "
                      "numbers as exact."],
        }

    notes: list[str] = []
    n_frames = max(int(meta.n_frames), 1)

    # ------------------------------------------------------------ calibration
    calib_cov = 0.0
    reproj_med = None
    method_mix: dict[str, int] = {}
    if homography is not None and len(homography):
        valid = homography["h00"].notna()
        calib_cov = float(valid.mean())
        method_mix = homography.loc[valid, "method"].value_counts().to_dict()
        direct = homography[valid & ~homography["method"].isin(["flow", "none"])]
        if len(direct):
            reproj_med = float(direct["reproj_error_px"].median())
    lv_calib = _worst(_grade(calib_cov, CALIB_COVERAGE),
                      _grade(reproj_med, CALIB_REPROJ_PX, higher_is_better=False))
    if calib_cov < CALIB_COVERAGE["medium"]:
        notes.append(f"Only {100 * calib_cov:.0f}% of frames have a pitch "
                     "calibration — positions outside those frames are missing.")
    flow_share = method_mix.get("flow", 0) / max(sum(method_mix.values()), 1)
    if flow_share > 0.85 and lv_calib != "low":
        notes.append("Calibration is mostly optical-flow bridging between "
                     "sparse direct solves — positional drift grows between anchors.")

    # ------------------------------------------------------------------- ball
    ball = tracking[tracking["entity_id"] == BALL_ID]
    ball_cov = float(ball["frame"].nunique() / n_frames)
    conf = ball["conf"].to_numpy(dtype=float)
    observed = meta.extras.get("ball_observed_frames")
    estimated = observed is None
    if estimated and len(conf):
        # Interpolated rows carry conf decayed to <=0.5x their neighbours
        # (see perception.detection.ball.interpolate_ball); half the q90 conf
        # separates observed from bridged rows well on the reference jobs.
        thr = 0.5 * float(np.quantile(conf, 0.9))
        observed = int(ball.loc[conf > thr, "frame"].nunique())
    observed_share = float(observed or 0) / n_frames
    lv_ball = _grade(observed_share, BALL_OBSERVED)
    if lv_ball != "high":
        notes.append(
            f"The ball was directly observed in only {100 * observed_share:.0f}% "
            "of frames (the rest are interpolated or missing) — possession, "
            "pass and xT figures inherit this uncertainty.")

    # --------------------------------------------------------------- tracking
    players = tracking[tracking["entity_id"] != BALL_ID]
    n_tracks = int(players["entity_id"].nunique())
    per_frame_med = float(players.groupby("frame").size().median()) if len(players) else 0.0
    track_len_med_s = 0.0
    if n_tracks and meta.fps > 0:
        track_len_med_s = float(players.groupby("entity_id").size().median() / meta.fps)
    lv_track = _grade(track_len_med_s, TRACK_LEN_S)
    if lv_track == "low":
        notes.append(f"Identities are short-lived (median track {track_len_med_s:.1f}s "
                     f"across {n_tracks} tracks) — per-player aggregates are unreliable.")
    if per_frame_med < 14:
        notes.append(f"Only ~{per_frame_med:.0f} players visible per frame "
                     "(broadcast framing) — team-shape and pitch-control metrics "
                     "describe the visible players.")

    # ------------------------------------------------------------------ teams
    sep = meta.extras.get("team_separability")
    lv_teams = _grade(float(sep) if sep is not None else None, TEAM_SEP)
    if lv_teams == "low":
        notes.append("Kit colours were hard to separate — team labels (and every "
                     "team-split metric) may be partially wrong.")

    if detector.startswith(("blob", "coco")):
        notes.append(f"Detection ran on the fallback stack ({detector}) — the "
                     "trained football detector was not available for this run.")

    overall = _worst(lv_calib, lv_ball, lv_track, lv_teams)
    return {
        "overall": overall,
        "is_cv": True,
        "levels": {"calibration": lv_calib, "ball": lv_ball,
                   "tracking": lv_track, "teams": lv_teams},
        "components": {
            "calibration_coverage": round(calib_cov, 3),
            "calibration_reproj_px_median": (round(reproj_med, 1)
                                             if reproj_med is not None else None),
            "calibration_method_mix": method_mix,
            "ball_frame_coverage": round(ball_cov, 3),
            "ball_observed_share": round(observed_share, 3),
            "ball_observed_is_estimated": estimated,
            "ball_conf_median": (round(float(np.median(conf)), 3) if len(conf) else None),
            "players_per_frame_median": per_frame_med,
            "n_player_tracks": n_tracks,
            "track_len_s_median": round(track_len_med_s, 1),
            "team_separability": (round(float(sep), 2) if sep is not None else None),
            "detector": detector,
        },
        "notes": notes,
    }


def assess_quality_from_store(store: ArtifactStore) -> dict:
    """Convenience wrapper reading the artifacts a job already has.

    Non-CV jobs (simulator ground truth, data-provider imports) are graded
    from meta alone — they may legitimately lack tracking/homography files.
    A CV job without a tracking table is genuinely broken and raises.
    """
    meta = store.load_meta()
    if not str(meta.source or "").startswith("video:"):
        return assess_quality(pd.DataFrame(), None, meta)
    tracking = store.load_tracking()
    homography = store.load_homography() if store.homography_path.exists() else None
    return assess_quality(tracking, homography, meta)
