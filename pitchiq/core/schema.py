"""The tracking table — PitchIQ's core artifact — and its parquet IO.

Schema (one row per entity per frame)::

    frame | timestamp | entity_id | class | team | jersey_no |
    x_pixel | y_pixel | x_pitch | y_pitch | conf

Plus a homography table (one row per frame) storing the flattened 3x3 matrix
and calibration quality. Both are persisted as parquet so Layers 2-3 rerun as
pure dataframe analysis without touching video.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TRACKING_COLUMNS: dict[str, str] = {
    "frame": "int32",
    "timestamp": "float64",
    "entity_id": "int32",
    "class": "category",
    "team": "category",
    "jersey_no": "Int16",  # nullable
    "x_pixel": "float32",
    "y_pixel": "float32",
    "x_pitch": "float32",
    "y_pitch": "float32",
    "conf": "float32",
}

HOMOGRAPHY_COLUMNS = (
    ["frame", "timestamp"]
    + [f"h{i}{j}" for i in range(3) for j in range(3)]
    + ["reproj_error_px", "method", "is_scene_cut"]
)

BALL_ID = -1  # the ball uses a reserved entity_id


def empty_tracking_table() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in TRACKING_COLUMNS.items()})
    return df


def validate_tracking_table(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes, order columns, and sanity-check the tracking table."""
    missing = set(TRACKING_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"tracking table missing columns: {sorted(missing)}")
    out = df[list(TRACKING_COLUMNS)].copy()
    for col, dtype in TRACKING_COLUMNS.items():
        out[col] = out[col].astype(dtype)
    if (out["frame"] < 0).any():
        raise ValueError("negative frame indices")
    return out.sort_values(["frame", "entity_id"]).reset_index(drop=True)


def save_tracking_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_tracking_table(df).to_parquet(path, index=False)


def load_tracking_table(path: str | Path) -> pd.DataFrame:
    return validate_tracking_table(pd.read_parquet(path))


def homographies_to_frame(records: list[dict]) -> pd.DataFrame:
    """Build the homography table from per-frame dicts with H (3x3) or None."""
    rows = []
    for rec in records:
        H = rec.get("H")
        flat = (np.full(9, np.nan) if H is None else np.asarray(H, dtype=np.float64).ravel())
        row = {"frame": rec["frame"], "timestamp": rec["timestamp"]}
        row.update({f"h{i}{j}": flat[i * 3 + j] for i in range(3) for j in range(3)})
        row["reproj_error_px"] = rec.get("reproj_error_px", np.nan)
        row["method"] = rec.get("method", "none")
        row["is_scene_cut"] = bool(rec.get("is_scene_cut", False))
        rows.append(row)
    return pd.DataFrame(rows, columns=HOMOGRAPHY_COLUMNS)


def save_homographies(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_homographies(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def homography_for_frame(hdf: pd.DataFrame, frame: int) -> np.ndarray | None:
    row = hdf.loc[hdf["frame"] == frame]
    if row.empty:
        return None
    vals = row.iloc[0][[f"h{i}{j}" for i in range(3) for j in range(3)]].to_numpy(dtype=np.float64)
    if np.isnan(vals).any():
        return None
    return vals.reshape(3, 3)


class MatchMeta:
    """Lightweight match metadata persisted as ``meta.json`` next to the table."""

    def __init__(
        self,
        fps: float,
        n_frames: int,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        team_names: dict[str, str] | None = None,
        kit_colors: dict[str, str] | None = None,
        attack_direction: dict[str, int] | None = None,
        source: str = "",
        extras: dict | None = None,
    ) -> None:
        self.fps = fps
        self.n_frames = n_frames
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.team_names = team_names or {"home": "Home", "away": "Away"}
        self.kit_colors = kit_colors or {"home": "#d62728", "away": "#1f77b4"}
        # +1: attacks positive-x. Keyed by team; second half flips sign implicitly
        # only if extras["halftime_frame"] is set.
        self.attack_direction = attack_direction or {"home": 1, "away": -1}
        self.source = source
        self.extras = extras or {}

    def to_dict(self) -> dict:
        return {
            "fps": self.fps,
            "n_frames": self.n_frames,
            "pitch_length": self.pitch_length,
            "pitch_width": self.pitch_width,
            "team_names": self.team_names,
            "kit_colors": self.kit_colors,
            "attack_direction": self.attack_direction,
            "source": self.source,
            "extras": self.extras,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MatchMeta":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**d)

    def attack_sign(self, team: str, frame: int) -> int:
        """Attacking direction of ``team`` at ``frame`` (+1 = positive-x)."""
        base = int(self.attack_direction.get(team, 1))
        halftime = self.extras.get("halftime_frame")
        if halftime is not None and frame >= int(halftime):
            return -base
        return base
