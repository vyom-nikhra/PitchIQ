"""Team-assignment tests: kit-colour clustering must separate real-ish kits.

The regression these guard against: on 576p broadcast footage the original
lightness-heavy signature collapsed nearly all players into one cluster
(observed ~20k vs ~2k rows). The chroma-first signature must keep white-kit
vs sky-blue-kit players in balanced, well-separated clusters.
"""

import numpy as np

from pitchiq.config import TeamsConfig
from pitchiq.core.types import EntityClass
from pitchiq.perception.teams import TeamAssigner
from pitchiq.perception.teams.assign import kit_signature

# BGR kit colours (OpenCV order)
WHITE_KIT = np.array([235, 235, 235], dtype=np.uint8)      # Real Madrid-ish
SKYBLUE_KIT = np.array([215, 170, 105], dtype=np.uint8)    # Man City-ish (BGR)
GRASS_BGR = np.array([55, 130, 60], dtype=np.uint8)        # pitch green


def _player_frame(kit_bgr: np.ndarray, box=(40, 80), seed=0) -> tuple[np.ndarray, np.ndarray]:
    """A grass frame with one kit-coloured torso; returns (frame, bbox)."""
    rng = np.random.default_rng(seed)
    w, h = box
    frame = np.tile(GRASS_BGR, (h + 20, w + 20, 1)).astype(np.uint8)
    frame = np.clip(frame + rng.integers(-8, 8, frame.shape), 0, 255).astype(np.uint8)
    x0, y0 = 10, 10
    # paint the torso region (top ~12-52% of the box) with the kit colour
    ty0, ty1 = y0 + int(0.12 * h), y0 + int(0.52 * h)
    tx0, tx1 = x0 + int(0.2 * w), x0 + int(0.8 * w)
    patch = np.tile(kit_bgr, (ty1 - ty0, tx1 - tx0, 1)).astype(np.int16)
    patch = np.clip(patch + rng.integers(-12, 12, patch.shape), 0, 255).astype(np.uint8)
    frame[ty0:ty1, tx0:tx1] = patch
    bbox = np.array([x0, y0, x0 + w, y0 + h], dtype=float)
    return frame, bbox


def test_kit_signature_separates_white_and_skyblue():
    """The raw signatures must be closer within a kit than across kits."""
    cfg = TeamsConfig()
    sigs = {"white": [], "blue": []}
    for i in range(6):
        fw, bw = _player_frame(WHITE_KIT, seed=i)
        fb, bb = _player_frame(SKYBLUE_KIT, seed=100 + i)
        from pitchiq.perception.teams.assign import torso_crop

        sigs["white"].append(kit_signature(torso_crop(fw, bw, cfg), cfg.min_non_grass_px))
        sigs["blue"].append(kit_signature(torso_crop(fb, bb, cfg), cfg.min_non_grass_px))
    white = np.stack(sigs["white"])
    blue = np.stack(sigs["blue"])
    intra = 0.5 * (np.std(white, axis=0).sum() + np.std(blue, axis=0).sum()) + 1e-6
    inter = np.linalg.norm(white.mean(0) - blue.mean(0))
    assert inter / intra > 2.0, f"kits not separable: inter/intra={inter / intra:.2f}"


def test_team_assigner_balanced_clusters():
    """Full assigner: 8 players/team must split into balanced HOME/AWAY."""
    cfg = TeamsConfig(min_samples_per_track=2)
    assigner = TeamAssigner(cfg)
    track_classes = {}
    n_per = 8
    for t in range(n_per):
        wid, bid = t, 100 + t
        track_classes[wid] = EntityClass.PLAYER
        track_classes[bid] = EntityClass.PLAYER
        for s in range(3):  # several samples per track
            fw, bw = _player_frame(WHITE_KIT, seed=t * 10 + s)
            fb, bb = _player_frame(SKYBLUE_KIT, seed=1000 + t * 10 + s)
            assigner.add_sample(wid, fw, bw)
            assigner.add_sample(bid, fb, bb)
    # give the two teams distinct mean-x so GK/side logic has data
    mean_x = {t: 30.0 for t in range(n_per)} | {100 + t: 75.0 for t in range(n_per)}
    result = assigner.finalize(track_classes, mean_x, pitch_length=105.0)

    from pitchiq.core.types import Team

    counts = {Team.HOME: 0, Team.AWAY: 0}
    for tid, team in result.team_of_track.items():
        if team in counts:
            counts[team] += 1
    # both teams populated and roughly balanced (each 8 players)
    assert counts[Team.HOME] > 0 and counts[Team.AWAY] > 0
    balance = min(counts.values()) / max(counts.values())
    assert balance > 0.5, f"clusters imbalanced: {counts}"
    assert result.separability > 1.5, f"low separability {result.separability:.2f}"
    # white and blue tracks must land in different clusters
    white_teams = {result.team_of_track[t] for t in range(n_per)}
    blue_teams = {result.team_of_track[100 + t] for t in range(n_per)}
    assert len(white_teams) == 1 and len(blue_teams) == 1
    assert white_teams != blue_teams


def test_tiny_boxes_rejected():
    """Boxes below the torso-height floor must contribute no sample."""
    cfg = TeamsConfig(min_torso_height_px=22)
    assigner = TeamAssigner(cfg)
    frame, _ = _player_frame(WHITE_KIT, box=(40, 80))
    tiny = np.array([10, 10, 24, 24], dtype=float)  # 14px tall < 22
    assigner.add_sample(1, frame, tiny)
    assert 1 not in assigner._samples or len(assigner._samples.get(1, [])) == 0
