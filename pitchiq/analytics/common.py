"""Shared helpers for analytics modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitchiq.core.schema import BALL_ID, MatchMeta

PERSON_CLASSES = ("player", "goalkeeper")


def players_only(df: pd.DataFrame, include_gk: bool = True) -> pd.DataFrame:
    """Rows of team players (optionally + GKs), with valid pitch coordinates."""
    classes = PERSON_CLASSES if include_gk else ("player",)
    out = df[df["class"].isin(classes) & df["team"].isin(["home", "away"])]
    return out.dropna(subset=["x_pitch", "y_pitch"])


def ball_series(df: pd.DataFrame) -> pd.DataFrame:
    """Ball rows indexed by frame with x/y (may have gaps)."""
    ball = df[df["entity_id"] == BALL_ID].set_index("frame").sort_index()
    return ball[["timestamp", "x_pitch", "y_pitch", "conf"]]


def attack_sign_series(meta: MatchMeta, frames: np.ndarray, team: str) -> np.ndarray:
    """Vector of ±1 attacking directions for ``team`` over ``frames``."""
    halftime = meta.extras.get("halftime_frame")
    base = int(meta.attack_direction.get(team, 1))
    signs = np.full(len(frames), base, dtype=int)
    if halftime is not None:
        signs[np.asarray(frames) >= int(halftime)] = -base
    return signs


def to_attacking_x(x: np.ndarray, signs: np.ndarray, length: float = 105.0) -> np.ndarray:
    """x-coordinate in the team's attacking frame (0 = own goal line)."""
    x = np.asarray(x, dtype=float)
    return np.where(signs > 0, x, length - x)


def to_attacking_xy(xy: np.ndarray, signs: np.ndarray, length: float = 105.0,
                    width: float = 68.0) -> np.ndarray:
    xy = np.asarray(xy, dtype=float).copy()
    neg = signs < 0
    xy[neg, 0] = length - xy[neg, 0]
    xy[neg, 1] = width - xy[neg, 1]
    return xy


def team_of_entities(df: pd.DataFrame) -> dict[int, str]:
    """entity_id -> team (modal), persons only."""
    persons = df[df["class"].isin(PERSON_CLASSES)]
    return persons.groupby("entity_id")["team"].agg(
        lambda s: s.mode().iat[0] if len(s.mode()) else "none"
    ).to_dict()


def class_of_entities(df: pd.DataFrame) -> dict[int, str]:
    return df.groupby("entity_id")["class"].agg(lambda s: s.mode().iat[0]).to_dict()


def jersey_of_entities(df: pd.DataFrame) -> dict[int, object]:
    ser = df.groupby("entity_id")["jersey_no"].agg(
        lambda s: int(s.mode().iat[0]) if s.notna().any() else None
    )
    return ser.to_dict()


def minutes_played(df: pd.DataFrame, fps: float) -> dict[int, float]:
    counts = df[df["class"].isin(PERSON_CLASSES)].groupby("entity_id")["frame"].nunique()
    return (counts / fps / 60.0).to_dict()


def player_label(entity_id: int, jersey: dict[int, object], team: dict[int, str],
                 team_names: dict[str, str] | None = None) -> str:
    """Human-readable label: '#10 (Home)' or 'id 37 (Away)'."""
    t = team.get(entity_id, "?")
    tname = (team_names or {}).get(t, t)
    j = jersey.get(entity_id)
    core = f"#{j}" if j is not None else f"id{entity_id}"
    return f"{core} ({tname})"
