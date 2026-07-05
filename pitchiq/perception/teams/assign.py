"""Team assignment: cluster kit-colour signatures into two teams + outliers.

Approach (config: ``teams``):

1. For sampled frames, crop each track's torso (bbox fractions that avoid
   shorts, socks and grass), suppress residual grass pixels by hue, and build a
   colour signature: grass-free LAB mean (lightness down-weighted for shadow
   robustness) + an 8-bin hue histogram.
2. Aggregate per track (median over samples — robust to occlusions).
3. K-Means (k=2) over outfield-player tracks = the two teams. Tracks far from
   both centroids are outliers: referees/goalkeepers when the detector could
   not separate those classes (COCO-fallback mode).
4. Goalkeepers get their *team* from geometry, not colour (their kit matches
   neither team): the GK defends the side where they spend most time, and the
   team defending that side is known from mean player positions.

Known limitations (documented in README): very similar kit colours can merge
clusters (flagged via a separability score in the result), and heavy
shadow/lighting splits are only partly absorbed by the L* down-weighting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from sklearn.cluster import KMeans

from pitchiq.config import TeamsConfig
from pitchiq.core.types import EntityClass, Team

log = logging.getLogger(__name__)

GRASS_HUE = (30, 95)  # HSV hue range treated as grass inside crops


def torso_crop(frame_bgr: np.ndarray, bbox: np.ndarray, cfg: TeamsConfig) -> np.ndarray | None:
    """Jersey region of a person bbox (fractions from config)."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    w, h = x2 - x1, y2 - y1
    min_h = getattr(cfg, "min_torso_height_px", 12)
    if w < 6 or h < min_h:
        return None
    cx1 = x1 + int(cfg.torso_inset * w)
    cx2 = x2 - int(cfg.torso_inset * w)
    cy1 = y1 + int(cfg.torso_top * h)
    cy2 = y1 + int(cfg.torso_bottom * h)
    H, W = frame_bgr.shape[:2]
    cx1, cy1 = max(0, cx1), max(0, cy1)
    cx2, cy2 = min(W, cx2), min(H, cy2)
    if cx2 - cx1 < 3 or cy2 - cy1 < 3:
        return None
    return frame_bgr[cy1:cy2, cx1:cx2]


def kit_signature(crop_bgr: np.ndarray, min_non_grass_px: int = 40) -> np.ndarray | None:
    """Chroma-first colour signature for K-Means team clustering.

    The discriminating signal between kits lives in the LAB *chroma* channels
    (a*, b*), not lightness: e.g. a white kit sits near neutral b* while a
    sky-blue kit has b* well below 128. L* mostly carries shadow/exposure
    variation across the pitch, so it is heavily down-weighted. a*/b* are
    centred on 128 (their neutral point) and amplified, and an 8-bin hue
    histogram over the non-grass torso pixels adds a saturation-robust cue.
    Returns ``None`` when too few non-grass pixels remain to trust the crop.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    not_grass = ~((hue >= GRASS_HUE[0]) & (hue <= GRASS_HUE[1]) & (sat > 40))
    if not_grass.sum() < min_non_grass_px:
        return None
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    px = lab[not_grass]
    mean_lab = px.mean(axis=0)
    hist = cv2.calcHist([hsv], [0], not_grass.astype(np.uint8) * 255, [8], [0, 180]).ravel()
    hist = hist / (hist.sum() + 1e-9)
    sig = np.concatenate([
        [0.15 * mean_lab[0],                 # lightness: down-weighted
         1.6 * (mean_lab[1] - 128.0),        # a* chroma, centred + amplified
         1.6 * (mean_lab[2] - 128.0)],       # b* chroma, centred + amplified
        32.0 * hist,
    ])
    return sig.astype(np.float32)


@dataclass
class TeamAssignmentResult:
    team_of_track: dict[int, Team]
    cls_override: dict[int, EntityClass] = field(default_factory=dict)
    kit_colors: dict[str, str] = field(default_factory=dict)  # hex, for viz
    separability: float = 0.0  # inter-centroid dist / mean intra dist; <1.5 is suspect
    notes: list[str] = field(default_factory=list)


class TeamAssigner:
    """Accumulates per-track kit samples online, clusters once at the end."""

    def __init__(self, cfg: TeamsConfig) -> None:
        self.cfg = cfg
        self._samples: dict[int, list[np.ndarray]] = {}
        self._crops_rgbmean: dict[int, list[np.ndarray]] = {}

    def add_sample(self, track_id: int, frame_bgr: np.ndarray, bbox: np.ndarray) -> None:
        crop = torso_crop(frame_bgr, bbox, self.cfg)
        if crop is None:
            return
        sig = kit_signature(crop, getattr(self.cfg, "min_non_grass_px", 40))
        if sig is None:
            return
        self._samples.setdefault(track_id, []).append(sig)
        self._crops_rgbmean.setdefault(track_id, []).append(
            crop.reshape(-1, 3).mean(axis=0)[::-1]  # BGR->RGB
        )

    # ------------------------------------------------------------------ fit
    def finalize(
        self,
        track_classes: dict[int, EntityClass],
        track_mean_x: dict[int, float] | None = None,
        pitch_length: float = 105.0,
    ) -> TeamAssignmentResult:
        """Cluster accumulated signatures into teams.

        ``track_mean_x``: mean pitch-x per track (may be None before
        calibration) — used for GK team assignment and referee heuristics.
        """
        notes: list[str] = []
        usable = {
            tid: np.median(np.stack(sigs), axis=0)
            for tid, sigs in self._samples.items()
            if len(sigs) >= self.cfg.min_samples_per_track
        }
        for tid, sigs in self._samples.items():  # short tracks: use what we have
            if tid not in usable and sigs:
                usable[tid] = np.median(np.stack(sigs), axis=0)

        team_of: dict[int, Team] = {}
        cls_override: dict[int, EntityClass] = {}
        if not usable:
            return TeamAssignmentResult(team_of, notes=["no kit samples collected"])

        ref_ids = {t for t, c in track_classes.items() if c == EntityClass.REFEREE}
        gk_ids = {t for t, c in track_classes.items() if c == EntityClass.GOALKEEPER}
        field_ids = [t for t in usable if t not in ref_ids and t not in gk_ids]
        if len(field_ids) < 4:
            notes.append("too few outfield tracks for team clustering")
            return TeamAssignmentResult({t: Team.NONE for t in usable}, notes=notes)

        X_raw = np.stack([usable[t] for t in field_ids])
        # Standardise (whiten) each signature dimension before K-Means. Without
        # this, a single large-scale dimension dominates the Euclidean split and
        # collapses one team into a tiny outlier cluster — which is exactly what
        # happened on 576p white-vs-sky-blue footage, where the teams differ
        # mostly in lightness (small absolute spread) rather than a large-scale
        # channel. Whitening lets K-Means find the true 2-team structure on
        # whichever axis carries the signal.
        mu = X_raw.mean(axis=0)
        sd = X_raw.std(axis=0) + 1e-6
        X = (X_raw - mu) / sd
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        labels = km.labels_
        d_own = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)
        intra = float(np.median(d_own) + 1e-6)
        inter = float(np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1]))
        separability = inter / intra
        if separability < 1.5:
            notes.append(
                f"kit colours poorly separable (score {separability:.2f}); team labels may be noisy"
            )

        # outliers: far from both centroids -> referee/GK candidates (COCO mode).
        # The threshold is scale-robust in the whitened space: a genuine team
        # member sits close to its centroid relative to the inter-centroid gap,
        # so requiring the distance to also exceed a fraction of that gap avoids
        # flagging normal members when clusters are tight.
        outlier_thresh = max(2.5 * intra, 0.6 * inter)
        detector_has_classes = bool(ref_ids or gk_ids)
        for i, tid in enumerate(field_ids):
            if d_own[i] > outlier_thresh and not detector_has_classes:
                team_of[tid] = Team.NONE
            else:
                team_of[tid] = Team.HOME if labels[i] == 0 else Team.AWAY

        outliers = [t for t in field_ids if team_of.get(t) == Team.NONE]
        if outliers and track_mean_x:
            self._classify_outliers(outliers, track_mean_x, pitch_length, team_of, cls_override, notes)
        elif outliers:
            for t in outliers:
                cls_override[t] = EntityClass.REFEREE
            notes.append(f"{len(outliers)} colour outliers marked referee (no positions available)")

        # explicit referee class
        for t in ref_ids:
            team_of[t] = Team.NONE

        # team sides from mean player x
        sides = self._team_sides(team_of, track_mean_x)
        for t in gk_ids:
            team_of[t] = self._gk_team(t, track_mean_x, sides, pitch_length)

        kit_colors = self._mean_kit_hex(team_of)
        return TeamAssignmentResult(team_of, cls_override, kit_colors, separability, notes)

    # ------------------------------------------------------------- helpers
    def _classify_outliers(self, outliers, track_mean_x, pitch_length, team_of, cls_override, notes):
        """COCO-fallback: decide referee vs goalkeeper for colour outliers by
        position: GKs live at the pitch ends, referees roam centrally."""
        sides = self._team_sides(team_of, track_mean_x)
        for t in outliers:
            x = track_mean_x.get(t)
            if x is None:
                cls_override[t] = EntityClass.REFEREE
                continue
            if x < 0.18 * pitch_length or x > 0.82 * pitch_length:
                cls_override[t] = EntityClass.GOALKEEPER
                team_of[t] = self._gk_team(t, track_mean_x, sides, pitch_length)
            else:
                cls_override[t] = EntityClass.REFEREE
        n_gk = sum(1 for c in cls_override.values() if c == EntityClass.GOALKEEPER)
        notes.append(f"outliers classified heuristically: {n_gk} GK, {len(outliers)-n_gk} referee")

    @staticmethod
    def _team_sides(team_of: dict[int, Team], track_mean_x: dict[int, float] | None) -> dict[Team, str]:
        """Which side ('left'/'right') each team defends, from mean positions."""
        if not track_mean_x:
            return {}
        xs = {Team.HOME: [], Team.AWAY: []}
        for t, team in team_of.items():
            if team in xs and t in track_mean_x and track_mean_x[t] is not None:
                xs[team].append(track_mean_x[t])
        if not xs[Team.HOME] or not xs[Team.AWAY]:
            return {}
        home_left = np.mean(xs[Team.HOME]) <= np.mean(xs[Team.AWAY])
        return {
            Team.HOME: "left" if home_left else "right",
            Team.AWAY: "right" if home_left else "left",
        }

    @staticmethod
    def _gk_team(tid, track_mean_x, sides, pitch_length) -> Team:
        if not track_mean_x or tid not in track_mean_x or not sides:
            return Team.NONE
        gk_side = "left" if track_mean_x[tid] < pitch_length / 2 else "right"
        for team, side in sides.items():
            if side == gk_side:
                return team
        return Team.NONE

    def _mean_kit_hex(self, team_of: dict[int, Team]) -> dict[str, str]:
        out = {}
        for team, key in ((Team.HOME, "home"), (Team.AWAY, "away")):
            rgb = [
                np.median(np.stack(self._crops_rgbmean[t]), axis=0)
                for t, tm in team_of.items()
                if tm == team and t in self._crops_rgbmean
            ]
            if rgb:
                r, g, b = np.median(np.stack(rgb), axis=0).astype(int)
                out[key] = f"#{r:02x}{g:02x}{b:02x}"
        return out
