"""Tactical team profiles driving the simulator's behaviour."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlayerStyle:
    """Per-player behaviour modifiers keyed by slot index (0..9).

    ``roam``: how far the player drifts from their slot (0..1)
    ``forward_bias``: extra push toward the opponent goal in possession
    ``press_eagerness``: how aggressively they leave shape to press
    ``speed``: max speed in m/s
    """

    roam: float = 0.35
    forward_bias: float = 0.0
    press_eagerness: float = 0.5
    speed: float = 7.4


@dataclass
class TeamProfile:
    name: str
    formation: str = "4-3-3"
    kit_color: str = "#d62728"
    line_height: float = 0.5      # 0 = very deep block, 1 = very high line
    width: float = 0.5            # compact <-> expansive in possession
    press_intensity: float = 0.5  # 0 = passive block, 1 = all-out press
    marking_scheme: str = "zonal"  # 'man' | 'zonal'
    possession_style: str = "short"  # 'short' | 'direct'
    player_styles: dict[int, PlayerStyle] = field(default_factory=dict)

    def style(self, slot: int) -> PlayerStyle:
        return self.player_styles.get(slot, PlayerStyle())


def demo_profiles() -> tuple["TeamProfile", "TeamProfile"]:
    """The bundled demo fixture: a pressing 4-3-3 vs a man-marking deep 4-4-2.

    Deliberately contrasting so every analytics module has signal to find:
    press vs block, short vs direct, zonal vs man, plus scripted player
    personalities (an attacking fullback, a deep-lying playmaker, a
    box-to-box runner) for role discovery to rediscover.
    """
    home = TeamProfile(
        name="Crimson City",
        formation="4-3-3",
        kit_color="#d62728",
        line_height=0.72,
        width=0.62,
        press_intensity=0.8,
        marking_scheme="zonal",
        possession_style="short",
        player_styles={
            0: PlayerStyle(roam=0.55, forward_bias=0.30, press_eagerness=0.5, speed=8.2),  # RB overlaps
            4: PlayerStyle(roam=0.30, forward_bias=-0.10, press_eagerness=0.4, speed=7.0),  # DM stays
            5: PlayerStyle(roam=0.65, forward_bias=0.25, press_eagerness=0.7, speed=7.9),  # B2B RCM
            8: PlayerStyle(roam=0.45, forward_bias=0.15, press_eagerness=0.9, speed=8.4),  # pressing ST
        },
    )
    away = TeamProfile(
        name="Azure United",
        formation="4-4-2",
        kit_color="#1f77b4",
        line_height=0.30,
        width=0.40,
        press_intensity=0.25,
        marking_scheme="man",
        possession_style="direct",
        player_styles={
            6: PlayerStyle(roam=0.55, forward_bias=0.10, press_eagerness=0.5, speed=7.6),  # LCM playmaker
            9: PlayerStyle(roam=0.40, forward_bias=0.20, press_eagerness=0.6, speed=8.6),  # quick LST
        },
    )
    return home, away
