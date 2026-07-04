"""Speed, distance, acceleration, sprints from projected positions.

Calibration jitter differentiates into huge phantom speeds, so positions are
Savitzky-Golay smoothed before differencing and speeds above a physiological
cap are masked (and excluded from distance). Output is both a per-frame
kinematics table (needed by pitch control and pressing) and per-player
aggregate profiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from pitchiq.config import KinematicsConfig
from pitchiq.analytics.common import PERSON_CLASSES


def compute_kinematics(df: pd.DataFrame, fps: float, cfg: KinematicsConfig) -> pd.DataFrame:
    """Per-frame smoothed positions and velocities for every person entity.

    Returns columns: frame, entity_id, x, y, vx, vy, speed, accel.
    Frames where the entity was unobserved are absent (no interpolation
    across long occlusions — downstream code must tolerate gaps).
    """
    persons = df[df["class"].isin(PERSON_CLASSES + ("referee",))].dropna(
        subset=["x_pitch", "y_pitch"]
    )
    window = max(5, int(round(cfg.smooth_window_s * fps)) | 1)  # odd
    out_parts: list[pd.DataFrame] = []
    for eid, g in persons.groupby("entity_id"):
        g = g.sort_values("frame")
        # bridge micro-gaps (<= 3 frames) so smoothing has continuity
        frames = g["frame"].to_numpy()
        x = g["x_pitch"].to_numpy(dtype=float)
        y = g["y_pitch"].to_numpy(dtype=float)
        if len(g) < window:
            continue
        # split into contiguous runs (gap > 3 frames breaks the run)
        breaks = np.where(np.diff(frames) > 3)[0] + 1
        for seg_idx in np.split(np.arange(len(frames)), breaks):
            if len(seg_idx) < window:
                continue
            f_seg = frames[seg_idx]
            # resample to every frame in the run (linear for tiny gaps)
            full = np.arange(f_seg[0], f_seg[-1] + 1)
            xi = np.interp(full, f_seg, x[seg_idx])
            yi = np.interp(full, f_seg, y[seg_idx])
            xs = savgol_filter(xi, window, 2)
            ys = savgol_filter(yi, window, 2)
            vx = np.gradient(xs) * fps
            vy = np.gradient(ys) * fps
            speed = np.hypot(vx, vy)
            # mask unphysical (kickoff teleports, calibration glitches)
            bad = speed > cfg.max_speed_mps
            speed_m = np.where(bad, np.nan, speed)
            accel = np.gradient(np.nan_to_num(speed_m, nan=0.0)) * fps
            out_parts.append(pd.DataFrame({
                "frame": full, "entity_id": eid,
                "x": xs, "y": ys, "vx": vx, "vy": vy,
                "speed": speed_m, "accel": accel,
            }))
    if not out_parts:
        return pd.DataFrame(columns=["frame", "entity_id", "x", "y", "vx", "vy", "speed", "accel"])
    return pd.concat(out_parts, ignore_index=True).sort_values(["frame", "entity_id"])


def player_movement_profiles(kin: pd.DataFrame, fps: float, cfg: KinematicsConfig) -> dict[int, dict]:
    """Aggregate per player: distance, top speed, sprints, high-intensity distance."""
    profiles: dict[int, dict] = {}
    dt = 1.0 / fps
    min_sprint_frames = max(1, int(cfg.min_sprint_duration_s * fps))
    for eid, g in kin.groupby("entity_id"):
        g = g.sort_values("frame")
        sp = g["speed"].to_numpy()
        valid = np.isfinite(sp)
        dist = float(np.nansum(sp[valid]) * dt)
        minutes = len(g) / fps / 60.0
        hi_mask = valid & (sp >= cfg.hi_speed_mps)
        sprint_mask = valid & (sp >= cfg.sprint_speed_mps)
        # sprint bouts: contiguous runs above threshold
        n_sprints = 0
        run = 0
        for m in sprint_mask:
            run = run + 1 if m else 0
            if run == min_sprint_frames:
                n_sprints += 1
        accel_vals = g["accel"].to_numpy()
        profiles[int(eid)] = {
            "minutes": round(minutes, 2),
            "distance_m": round(dist, 1),
            "distance_per_min_m": round(dist / minutes, 1) if minutes > 0 else 0.0,
            "top_speed_mps": round(float(np.nanmax(sp)) if valid.any() else 0.0, 2),
            "avg_speed_mps": round(float(np.nanmean(sp)) if valid.any() else 0.0, 2),
            "hi_distance_m": round(float(np.nansum(sp[hi_mask]) * dt), 1),
            "n_sprints": int(n_sprints),
            "sprints_per_min": round(n_sprints / minutes, 2) if minutes > 0 else 0.0,
            "accel_p90": round(float(np.nanpercentile(np.abs(accel_vals), 90)), 2)
            if len(accel_vals) else 0.0,
            "direction_changes_per_min": _direction_changes_per_min(g, fps),
        }
    return profiles


def _direction_changes_per_min(g: pd.DataFrame, fps: float) -> float:
    """Heading reversals > 60° while moving — an agility/style signal."""
    vx = g["vx"].to_numpy()
    vy = g["vy"].to_numpy()
    speed = np.hypot(vx, vy)
    moving = speed > 1.5
    heading = np.arctan2(vy, vx)
    dh = np.abs(np.diff(heading))
    dh = np.minimum(dh, 2 * np.pi - dh)
    changes = int(np.sum((dh > np.radians(60)) & moving[1:]))
    minutes = len(g) / fps / 60.0
    return round(changes / minutes, 2) if minutes > 0 else 0.0
