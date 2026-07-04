"""Shared fixtures. The short simulated match is session-scoped: several test
modules exercise analytics against it."""

from __future__ import annotations

import pytest

from pitchiq.config import load_config
from pitchiq.core.pitch import Pitch


@pytest.fixture(scope="session")
def pitch() -> Pitch:
    return Pitch()


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def sim_result():
    """A 2x30s simulated match with ground truth (deterministic seed)."""
    from pitchiq.demo.simulate import simulate_demo_match

    cfg = load_config(overrides={"simulator": {"half_minutes": 0.5, "seed": 11}})
    return simulate_demo_match(cfg.simulator)
