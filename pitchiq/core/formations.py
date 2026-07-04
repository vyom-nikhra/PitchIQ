"""Canonical formation templates shared by the simulator, formation detection
and role naming.

Slots are normalised to a unit pitch attacking left→right: x in [0,1] is
depth (0 = own goal), y in [0,1] is width (0 = right touchline from the
attacking team's view... we simply use pitch-y). Goalkeeper excluded — all
templates have exactly 10 outfield slots with human position labels.
"""

from __future__ import annotations

import numpy as np

# name -> list of (x, y, label) for the 10 outfield players
FORMATIONS: dict[str, list[tuple[float, float, str]]] = {
    "4-4-2": [
        (0.18, 0.18, "RB"), (0.15, 0.40, "RCB"), (0.15, 0.60, "LCB"), (0.18, 0.82, "LB"),
        (0.45, 0.15, "RM"), (0.42, 0.40, "RCM"), (0.42, 0.60, "LCM"), (0.45, 0.85, "LM"),
        (0.72, 0.42, "RST"), (0.72, 0.58, "LST"),
    ],
    "4-3-3": [
        (0.18, 0.18, "RB"), (0.15, 0.40, "RCB"), (0.15, 0.60, "LCB"), (0.18, 0.82, "LB"),
        (0.38, 0.50, "DM"), (0.50, 0.32, "RCM"), (0.50, 0.68, "LCM"),
        (0.72, 0.15, "RW"), (0.78, 0.50, "ST"), (0.72, 0.85, "LW"),
    ],
    "4-2-3-1": [
        (0.18, 0.18, "RB"), (0.15, 0.40, "RCB"), (0.15, 0.60, "LCB"), (0.18, 0.82, "LB"),
        (0.38, 0.40, "RDM"), (0.38, 0.60, "LDM"),
        (0.58, 0.15, "RAM"), (0.60, 0.50, "CAM"), (0.58, 0.85, "LAM"),
        (0.78, 0.50, "ST"),
    ],
    "4-5-1": [
        (0.18, 0.18, "RB"), (0.15, 0.40, "RCB"), (0.15, 0.60, "LCB"), (0.18, 0.82, "LB"),
        (0.42, 0.12, "RM"), (0.40, 0.35, "RCM"), (0.38, 0.50, "CM"), (0.40, 0.65, "LCM"), (0.42, 0.88, "LM"),
        (0.72, 0.50, "ST"),
    ],
    "3-5-2": [
        (0.15, 0.28, "RCB"), (0.13, 0.50, "CB"), (0.15, 0.72, "LCB"),
        (0.45, 0.08, "RWB"), (0.42, 0.35, "RCM"), (0.40, 0.50, "CM"), (0.42, 0.65, "LCM"), (0.45, 0.92, "LWB"),
        (0.72, 0.42, "RST"), (0.72, 0.58, "LST"),
    ],
    "3-4-3": [
        (0.15, 0.28, "RCB"), (0.13, 0.50, "CB"), (0.15, 0.72, "LCB"),
        (0.45, 0.12, "RWB"), (0.42, 0.40, "RCM"), (0.42, 0.60, "LCM"), (0.45, 0.88, "LWB"),
        (0.72, 0.18, "RW"), (0.76, 0.50, "ST"), (0.72, 0.82, "LW"),
    ],
    "5-3-2": [
        (0.20, 0.08, "RWB"), (0.14, 0.30, "RCB"), (0.12, 0.50, "CB"), (0.14, 0.70, "LCB"), (0.20, 0.92, "LWB"),
        (0.42, 0.32, "RCM"), (0.40, 0.50, "CM"), (0.42, 0.68, "LCM"),
        (0.70, 0.42, "RST"), (0.70, 0.58, "LST"),
    ],
    "5-4-1": [
        (0.20, 0.08, "RWB"), (0.14, 0.30, "RCB"), (0.12, 0.50, "CB"), (0.14, 0.70, "LCB"), (0.20, 0.92, "LWB"),
        (0.42, 0.15, "RM"), (0.40, 0.40, "RCM"), (0.40, 0.60, "LCM"), (0.42, 0.85, "LM"),
        (0.70, 0.50, "ST"),
    ],
    "4-1-4-1": [
        (0.18, 0.18, "RB"), (0.15, 0.40, "RCB"), (0.15, 0.60, "LCB"), (0.18, 0.82, "LB"),
        (0.32, 0.50, "DM"),
        (0.52, 0.15, "RM"), (0.50, 0.38, "RCM"), (0.50, 0.62, "LCM"), (0.52, 0.85, "LM"),
        (0.75, 0.50, "ST"),
    ],
    "4-4-1-1": [
        (0.18, 0.18, "RB"), (0.15, 0.40, "RCB"), (0.15, 0.60, "LCB"), (0.18, 0.82, "LB"),
        (0.44, 0.15, "RM"), (0.42, 0.40, "RCM"), (0.42, 0.60, "LCM"), (0.44, 0.85, "LM"),
        (0.60, 0.50, "SS"), (0.76, 0.50, "ST"),
    ],
}

#: broad family of each slot label, used by role naming
LABEL_FAMILY = {
    "RB": "fullback", "LB": "fullback", "RWB": "fullback", "LWB": "fullback",
    "RCB": "centre-back", "LCB": "centre-back", "CB": "centre-back",
    "DM": "defensive-mid", "RDM": "defensive-mid", "LDM": "defensive-mid",
    "CM": "central-mid", "RCM": "central-mid", "LCM": "central-mid",
    "CAM": "attacking-mid", "SS": "attacking-mid", "RAM": "wide-mid", "LAM": "wide-mid",
    "RM": "wide-mid", "LM": "wide-mid",
    "RW": "winger", "LW": "winger",
    "ST": "striker", "RST": "striker", "LST": "striker",
}


def formation_slots(name: str, length: float = 105.0, width: float = 68.0,
                    attack_sign: int = 1) -> np.ndarray:
    """Template slots in pitch metres for a team attacking ``attack_sign`` x.

    Returns (10, 2) array ordered as in the template definition.
    """
    if name not in FORMATIONS:
        raise KeyError(f"unknown formation {name}; known: {sorted(FORMATIONS)}")
    slots = np.array([(x, y) for x, y, _ in FORMATIONS[name]], dtype=float)
    xy = np.empty_like(slots)
    if attack_sign >= 0:
        xy[:, 0] = slots[:, 0] * length
        xy[:, 1] = slots[:, 1] * width
    else:
        xy[:, 0] = (1.0 - slots[:, 0]) * length
        xy[:, 1] = (1.0 - slots[:, 1]) * width
    return xy


def formation_labels(name: str) -> list[str]:
    return [lab for _, _, lab in FORMATIONS[name]]
