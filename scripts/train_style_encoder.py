"""Train the contrastive style encoder (6.1b) on simulated matches.

Positive pairs = the same player's phase-conditioned heatmap stack from the
two different halves of a match; negatives = other players in the batch
(SimCLR NT-Xent). Simulated matches with varied seeds/profiles give an
unlimited licence-clean corpus; pass ``--jobs-dir`` to additionally harvest
pairs from processed real matches.

CPU-friendly (~2-4 min for the default settings). Output: torchscript
weights consumed automatically when ``embeddings.learned.weights`` points at
them.

Usage:
    python scripts/train_style_encoder.py --matches 8 --epochs 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def half_features(cfg, sim_result, half: int):
    """Player features computed on one half of a simulated match."""
    from pitchiq.analytics import kinematics as kin_mod
    from pitchiq.analytics import phases as phases_mod
    from pitchiq.analytics import possession as poss_mod
    from pitchiq.intelligence.features import extract_player_features

    df = sim_result.tracking
    meta = sim_result.meta
    halftime = meta.extras["halftime_frame"]
    mask = df.frame < halftime if half == 0 else df.frame >= halftime
    sub = df[mask].copy()
    if half == 1:  # rebase frames so kinematics windows behave
        sub["frame"] = sub["frame"] - halftime
        meta = type(meta)(**{**meta.to_dict(), "extras": {}})
    kin = kin_mod.compute_kinematics(sub, meta.fps, cfg.kinematics)
    poss = poss_mod.compute_possession(sub, meta.fps, cfg.possession)
    ph = phases_mod.segment_phases(sub, poss, meta, cfg.phases)
    import pandas as pd

    return extract_player_features(sub, kin, poss, pd.DataFrame(), ph, {},
                                   meta, cfg.embeddings, min_minutes=0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--half-minutes", type=float, default=1.5)
    ap.add_argument("--out", default="weights/style_encoder.pt")
    args = ap.parse_args()

    from pitchiq.config import load_config
    from pitchiq.demo.profiles import TeamProfile, demo_profiles
    from pitchiq.demo.simulate import MatchSimulator
    from pitchiq.intelligence.encoder import heatmap_stack, train_style_encoder

    rng = np.random.default_rng(0)
    formations = ["4-3-3", "4-4-2", "4-2-3-1", "3-5-2", "4-5-1", "3-4-3"]
    pairs = []
    for m in range(args.matches):
        cfg = load_config(overrides={"simulator": {
            "half_minutes": args.half_minutes, "seed": 100 + m}})
        home, away = demo_profiles()
        home = TeamProfile(name="H", formation=str(rng.choice(formations)),
                           line_height=float(rng.uniform(0.25, 0.8)),
                           press_intensity=float(rng.uniform(0.2, 0.9)),
                           marking_scheme=str(rng.choice(["man", "zonal"])),
                           possession_style=str(rng.choice(["short", "direct"])))
        away = TeamProfile(name="A", formation=str(rng.choice(formations)),
                           line_height=float(rng.uniform(0.25, 0.8)),
                           press_intensity=float(rng.uniform(0.2, 0.9)),
                           marking_scheme=str(rng.choice(["man", "zonal"])),
                           possession_style=str(rng.choice(["short", "direct"])))
        sim = MatchSimulator(cfg.simulator, home, away).run()
        f0 = half_features(cfg, sim, 0)
        f1 = half_features(cfg, sim, 1)
        for eid in set(f0) & set(f1):
            try:
                pairs.append((heatmap_stack(f0[eid]), heatmap_stack(f1[eid])))
            except ValueError:
                continue
        print(f"match {m + 1}/{args.matches}: {len(pairs)} pairs total")

    diag = train_style_encoder(pairs, dim=args.dim, epochs=args.epochs,
                               out_path=args.out)
    print("trained:", diag)


if __name__ == "__main__":
    main()
