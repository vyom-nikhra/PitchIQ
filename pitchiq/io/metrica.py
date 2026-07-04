"""Metrica Sports open tracking-data loader.

Metrica publishes anonymised full-match tracking (25 fps, both teams + ball)
at https://github.com/metrica-sports/sample-data. We use it to validate
homography-derived positions against professional tracking (positional error
in metres) and as a real-data source for the analytics layers.

Metrica coordinates are normalised to [0,1] x [0,1] with (0,0) top-left;
we convert to PitchIQ metres (x right, y up on a 105 x 68 pitch).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pitchiq.core.schema import BALL_ID, MatchMeta, validate_tracking_table
from pitchiq.io.errors import DatasetUnavailable

METRICA_URL = "https://github.com/metrica-sports/sample-data"


def _read_metrica_csv(path: Path) -> pd.DataFrame:
    """Parse Metrica's 3-row-header wide CSV into tidy (frame, player, x, y)."""
    raw = pd.read_csv(path, skiprows=2)
    header = pd.read_csv(path, nrows=2, header=None)
    player_names = header.iloc[1].tolist()  # 'Player11', ..., 'Ball'
    cols = list(raw.columns)
    tidy_rows = []
    # columns: Period, Frame, Time [s], then x,y pairs
    for i in range(3, len(cols) - 1, 2):
        name = str(player_names[i]) if i < len(player_names) else cols[i]
        sub = raw[[cols[0], cols[1], cols[2], cols[i], cols[i + 1]]].copy()
        sub.columns = ["period", "frame", "time_s", "x", "y"]
        sub["player"] = name.strip()
        tidy_rows.append(sub)
    return pd.concat(tidy_rows, ignore_index=True)


def load_metrica_match(
    data_dir: str | Path,
    game: str = "Sample_Game_1",
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> tuple[pd.DataFrame, MatchMeta]:
    """Load a Metrica sample game into a PitchIQ tracking table.

    ``data_dir`` must contain ``<game>/<game>_RawTrackingData_{Home,Away}_Team.csv``
    (the layout of the metrica-sports/sample-data repo's ``data`` folder).
    """
    data_dir = Path(data_dir)
    home_csv = data_dir / game / f"{game}_RawTrackingData_Home_Team.csv"
    away_csv = data_dir / game / f"{game}_RawTrackingData_Away_Team.csv"
    if not home_csv.exists() or not away_csv.exists():
        raise DatasetUnavailable(
            f"Metrica sample data not found under {data_dir}/{game}. "
            f"Clone {METRICA_URL} and point data_dir at its 'data' folder."
        )

    frames = []
    for team, csv_path in (("home", home_csv), ("away", away_csv)):
        tidy = _read_metrica_csv(csv_path)
        is_ball = tidy["player"].str.lower() == "ball"
        players = tidy[~is_ball].copy()
        players["entity_id"] = (
            players["player"].str.extract(r"(\d+)").astype(float).fillna(0).astype(int)
            + (0 if team == "home" else 100)
        )
        players["class"] = "player"
        players["team"] = team
        frames.append(players)
        if team == "home":  # ball appears in both files; take one
            ball = tidy[is_ball].copy()
            ball["entity_id"] = BALL_ID
            ball["class"] = "ball"
            ball["team"] = "none"
            frames.append(ball)

    df = pd.concat(frames, ignore_index=True).dropna(subset=["x", "y"])
    # Metrica normalised -> metres; flip y so y grows upward
    df["x_pitch"] = df["x"].astype(float) * pitch_length
    df["y_pitch"] = (1.0 - df["y"].astype(float)) * pitch_width

    fps = 25.0
    out = pd.DataFrame(
        {
            "frame": df["frame"].astype(int) - int(df["frame"].min()),
            "timestamp": df["time_s"].astype(float),
            "entity_id": df["entity_id"].astype(int),
            "class": df["class"],
            "team": df["team"],
            "jersey_no": df["player"].str.extract(r"(\d+)")[0].astype("Int16"),
            "x_pixel": np.float32(np.nan),
            "y_pixel": np.float32(np.nan),
            "x_pitch": df["x_pitch"].astype("float32"),
            "y_pitch": df["y_pitch"].astype("float32"),
            "conf": np.float32(1.0),
        }
    )
    meta = MatchMeta(
        fps=fps,
        n_frames=int(out["frame"].max()) + 1,
        team_names={"home": "Metrica Home", "away": "Metrica Away"},
        source=f"metrica:{game}",
    )
    return validate_tracking_table(out), meta
