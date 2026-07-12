"""ByteTrack multi-object tracker with BoT-SORT-style extensions.

Core algorithm (Zhang et al., 2022): two-stage association — confident
detections first (IoU + optional appearance), then low-score detections
against still-unmatched tracks, which is what keeps IDs alive through the
partial occlusions that plague football broadcast footage.

Extensions for panning broadcast cameras:
* optional appearance embeddings blended into the association cost
  (:mod:`pitchiq.perception.tracking.appearance`), which also re-links tracks
  after long occlusions,
* camera-motion compensation: predicted track boxes are warped by the
  frame-to-frame affine estimated from background optical flow before
  association (:mod:`pitchiq.perception.tracking.camera_motion`).

Implemented self-contained (numpy + scipy only) so tracking is unit-testable
without torch and swappable for ultralytics' built-in trackers if desired.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from pitchiq.config import TrackingConfig
from pitchiq.core.types import Detection, EntityClass, iou_matrix
from pitchiq.perception.tracking.kalman import KalmanBoxFilter, cxcyah_to_xyxy, xyxy_to_cxcyah

_shared_kf = KalmanBoxFilter()


class TrackState(Enum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class STrack:
    """One tracked entity: Kalman state + class votes + appearance memory."""

    _next_id = 1

    def __init__(self, det: Detection) -> None:
        self.mean, self.cov = _shared_kf.initiate(xyxy_to_cxcyah(det.bbox))
        self.state = TrackState.NEW
        self.track_id = 0  # assigned on activation
        self.score = det.conf
        self.cls_votes: dict[EntityClass, float] = {det.cls: det.conf}
        self.feature: np.ndarray | None = None
        self.alpha_feat = 0.9
        if det.feature is not None:
            self.feature = det.feature / (np.linalg.norm(det.feature) + 1e-9)
        self.frames_since_update = 0
        self.hits = 1
        self.start_frame = -1
        self.last_frame = -1

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def next_id(cls) -> int:
        i = cls._next_id
        cls._next_id += 1
        return i

    @classmethod
    def reset_ids(cls) -> None:
        cls._next_id = 1

    def activate(self, frame_id: int) -> None:
        self.track_id = STrack.next_id()
        self.state = TrackState.TRACKED
        self.start_frame = frame_id
        self.last_frame = frame_id

    def predict(self) -> None:
        mean = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean[7] = 0.0  # freeze height velocity while lost
        self.mean, self.cov = _shared_kf.predict(mean, self.cov)
        self.frames_since_update += 1

    def update(self, det: Detection, frame_id: int) -> None:
        self.mean, self.cov = _shared_kf.update(self.mean, self.cov, xyxy_to_cxcyah(det.bbox))
        self.state = TrackState.TRACKED
        self.score = det.conf
        self.hits += 1
        self.frames_since_update = 0
        self.last_frame = frame_id
        self.cls_votes[det.cls] = self.cls_votes.get(det.cls, 0.0) + det.conf
        if det.feature is not None:
            f = det.feature / (np.linalg.norm(det.feature) + 1e-9)
            if self.feature is None:
                self.feature = f
            else:
                self.feature = self.alpha_feat * self.feature + (1 - self.alpha_feat) * f
                self.feature /= np.linalg.norm(self.feature) + 1e-9

    def mark_lost(self) -> None:
        if self.state == TrackState.TRACKED:
            self.state = TrackState.LOST

    def apply_camera_motion(self, A: np.ndarray) -> None:
        """Warp the predicted box center by a 2x3 affine (camera compensation)."""
        cx, cy = self.mean[0], self.mean[1]
        pt = A @ np.array([cx, cy, 1.0])
        scale = float(np.sqrt(abs(np.linalg.det(A[:, :2])) + 1e-12))
        self.mean[0], self.mean[1] = pt[0], pt[1]
        self.mean[3] *= scale

    # ------------------------------------------------------------ accessors
    @property
    def bbox(self) -> np.ndarray:
        return cxcyah_to_xyxy(self.mean)

    @property
    def entity_class(self) -> EntityClass:
        return max(self.cls_votes, key=self.cls_votes.get)


def _assign(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment gated by ``cost <= thresh``.

    Returns (matches, unmatched_rows, unmatched_cols).
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    gated = np.where(cost > thresh, thresh + 1e5, cost)
    rows, cols = linear_sum_assignment(gated)
    matches, um_rows, um_cols = [], set(range(cost.shape[0])), set(range(cost.shape[1]))
    for r, c in zip(rows, cols):
        if cost[r, c] <= thresh:
            matches.append((r, c))
            um_rows.discard(r)
            um_cols.discard(c)
    return matches, sorted(um_rows), sorted(um_cols)


class ByteTracker:
    """Frame-by-frame tracker for person-type entities.

    The ball is deliberately excluded — it gets a dedicated selector
    (:mod:`pitchiq.perception.detection.ball`) because box IoU association is
    the wrong tool for a tiny fast object.
    """

    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        self.tracked: list[STrack] = []
        self.lost: list[STrack] = []
        self.frame_id = -1
        self._H = None
        self._dt = None
        self._prev_H = None            # last valid homography (pre-cut projection)
        self._cut_stash: list[dict] = []  # identities awaiting cross-cut re-ID
        STrack.reset_ids()

    def _cost(self, tracks: list[STrack], dets: list[Detection], use_appearance: bool) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)), dtype=np.float32)
        iou = iou_matrix(
            np.stack([t.bbox for t in tracks]), np.stack([d.bbox for d in dets])
        )
        cost = 1.0 - iou
        w = self.cfg.appearance.weight
        if use_appearance and self.cfg.appearance.enabled and w > 0:
            feats_ok = all(d.feature is not None for d in dets) and all(
                t.feature is not None for t in tracks
            )
            if feats_ok:
                tf = np.stack([t.feature for t in tracks])
                df = np.stack([d.feature / (np.linalg.norm(d.feature) + 1e-9) for d in dets])
                app_cost = (1.0 - tf @ df.T) / 2.0  # cosine distance in [0,1]
                # appearance only refines plausible spatial matches
                app_cost = np.where(cost < 0.95, app_cost, 1.0)
                cost = (1 - w) * cost + w * app_cost
        cost = self._apply_velocity_gate(cost, tracks, dets)
        return cost.astype(np.float32)

    def _apply_velocity_gate(self, cost, tracks, dets):
        """Set impossible (track, det) pairs to a rejecting cost using the
        pitch-space implied speed. No-op without a homography."""
        H = getattr(self, "_H", None)
        if H is None or not getattr(self, "_dt", None):
            return cost
        from pitchiq.core.geometry import apply_homography

        max_v = self.cfg.max_assoc_speed_mps
        det_foot = np.array([[(d.bbox[0] + d.bbox[2]) / 2.0, d.bbox[3]] for d in dets])
        trk_foot = np.array([getattr(t, "_gate_foot", ((t.bbox[0] + t.bbox[2]) / 2.0, t.bbox[3]))
                             for t in tracks])
        dp = apply_homography(H, det_foot)          # (D, 2) pitch metres
        tp = apply_homography(H, trk_foot)          # (T, 2)
        if not (np.all(np.isfinite(dp)) and np.all(np.isfinite(tp))):
            return cost
        dist = np.linalg.norm(tp[:, None, :] - dp[None, :, :], axis=2)  # (T, D) metres
        dt_eff = np.array([max(getattr(t, "_gate_dt", self._dt) or self._dt, self._dt)
                           for t in tracks])[:, None]
        speed = dist / np.maximum(dt_eff, 1e-6)
        # allow a small slack for calibration jitter near the boundary
        cost = cost.copy()
        cost[speed > max_v] = 1e5
        return cost

    # ------------------------------------------------------ cross-cut re-ID
    def _project_foot(self, t: STrack, H) -> np.ndarray | None:
        if H is None:
            return None
        from pitchiq.core.geometry import apply_homography

        x1, y1, x2, y2 = t.bbox
        p = apply_homography(H, [[(x1 + x2) / 2.0, y2]])[0]
        return p if np.all(np.isfinite(p)) else None

    def _stash_for_reid(self) -> None:
        """A scene cut invalidates every pixel-space state. Park all active
        identities (appearance + last pitch position via the pre-cut
        homography) so post-cut tracks can claim them back."""
        for t in self.tracked + self.lost:
            if t.track_id > 0:
                self._cut_stash.append(dict(
                    track=t, pitch=self._project_foot(t, self._prev_H),
                    frame=self.frame_id))
        self.tracked, self.lost = [], []

    def _try_reid(self) -> None:
        """Match tracks activated THIS frame against stashed identities:
        appearance cosine distance, gated by elapsed-time-aware pitch
        distance (a player can only have moved so far during the cutaway)."""
        if not self._cut_stash:
            return
        dt = self._dt or 0.04
        horizon_f = self.cfg.reid_horizon_s / dt
        self._cut_stash = [s for s in self._cut_stash
                           if self.frame_id - s["frame"] <= horizon_f]
        fresh = [t for t in self.tracked
                 if t.track_id > 0 and t.start_frame == self.frame_id]
        if not self._cut_stash or not fresh:
            return
        cost = np.ones((len(self._cut_stash), len(fresh)), dtype=np.float32)
        for i, s in enumerate(self._cut_stash):
            old: STrack = s["track"]
            elapsed_s = (self.frame_id - s["frame"]) * dt
            for j, new in enumerate(fresh):
                if old.feature is None or new.feature is None:
                    continue
                app = float((1.0 - old.feature @ new.feature) / 2.0)
                if s["pitch"] is not None:
                    p_new = self._project_foot(new, self._H)
                    if p_new is not None:
                        max_d = self.cfg.reid_base_radius_m + 5.0 * elapsed_s
                        if float(np.linalg.norm(p_new - s["pitch"])) > max_d:
                            continue  # too far to be the same player
                cost[i, j] = app
        matches, _, _ = _assign(cost, self.cfg.reid_appearance_thresh)
        claimed = []
        for si, fj in matches:
            old, new = self._cut_stash[si]["track"], fresh[fj]
            new.track_id = old.track_id
            for c, v in old.cls_votes.items():
                new.cls_votes[c] = new.cls_votes.get(c, 0.0) + v
            if old.feature is not None and new.feature is not None:
                f = 0.5 * old.feature + 0.5 * new.feature
                new.feature = f / (np.linalg.norm(f) + 1e-9)
            claimed.append(si)
        self._cut_stash = [s for i, s in enumerate(self._cut_stash)
                           if i not in claimed]

    def update(
        self, detections: list[Detection], camera_affine: np.ndarray | None = None,
        homography: np.ndarray | None = None, dt: float | None = None,
        scene_cut: bool = False,
    ) -> list[STrack]:
        """Advance one frame; returns currently tracked (active) tracks.

        ``homography`` (pixel→pitch) and ``dt`` (seconds/frame) enable the
        physical-plausibility gate: an association is rejected outright when it
        would require the entity to cover more real-pitch distance than a
        player can (``cfg.max_assoc_speed_mps``), measured from the track's
        last *observed* foot point to the candidate detection. This stops
        identity teleports at the source rather than only masking them later
        in the analytics kinematics. The gate is skipped when no homography is
        available for the frame (falls back to pixel IoU + Kalman).
        """
        self.frame_id += 1
        self._H = homography
        self._dt = dt
        if scene_cut and self.cfg.cross_cut_reid:
            self._stash_for_reid()
        dets = [d for d in detections if d.cls != EntityClass.BALL]
        high = [d for d in dets if d.conf >= self.cfg.high_thresh]
        low = [d for d in dets if self.cfg.low_thresh <= d.conf < self.cfg.high_thresh]

        for t in self.tracked + self.lost:
            # snapshot the last observed foot point + age BEFORE prediction so
            # the velocity gate measures real displacement over real elapsed time
            x1, y1, x2, y2 = t.bbox
            t._gate_foot = ((x1 + x2) / 2.0, y2)
            t._gate_dt = (t.frames_since_update + 1) * (dt or 0.0)
            t.predict()
            if camera_affine is not None:
                t.apply_camera_motion(camera_affine)

        confirmed = [t for t in self.tracked if t.state == TrackState.TRACKED]
        unconfirmed = [t for t in self.tracked if t.state == TrackState.NEW]

        # --- stage 1: confirmed + lost vs high-confidence detections
        pool = confirmed + self.lost
        cost = self._cost(pool, high, use_appearance=True)
        matches, um_tracks, um_dets = _assign(cost, self.cfg.match_thresh)
        for ti, di in matches:
            pool[ti].update(high[di], self.frame_id)

        # --- stage 2: remaining *confirmed* tracks vs low-confidence detections
        remain = [pool[i] for i in um_tracks if pool[i].state == TrackState.TRACKED]
        cost2 = self._cost(remain, low, use_appearance=False)
        matches2, um_tracks2, _ = _assign(cost2, self.cfg.second_match_thresh)
        for ti, di in matches2:
            remain[ti].update(low[di], self.frame_id)
        for i in um_tracks2:
            remain[i].mark_lost()
        # lost tracks that stayed unmatched remain lost (aged below)

        # --- unconfirmed tracks vs leftover high detections
        leftover_high = [high[i] for i in um_dets]
        cost3 = self._cost(unconfirmed, leftover_high, use_appearance=False)
        matches3, um_unconf, um_dets3 = _assign(cost3, self.cfg.unconfirmed_match_thresh)
        for ti, di in matches3:
            unconfirmed[ti].update(leftover_high[di], self.frame_id)
            if unconfirmed[ti].hits >= 2 and unconfirmed[ti].track_id == 0:
                unconfirmed[ti].activate(self.frame_id)
        removed_unconf = {id(unconfirmed[i]) for i in um_unconf}

        # --- births
        new_tracks: list[STrack] = []
        for di in um_dets3:
            d = leftover_high[di]
            if d.conf >= self.cfg.new_track_thresh:
                t = STrack(d)
                if self.frame_id == 0:  # first frame: trust immediately
                    t.activate(self.frame_id)
                new_tracks.append(t)

        # --- book-keeping
        next_tracked, next_lost = [], []
        for t in pool + unconfirmed + new_tracks:
            if id(t) in removed_unconf:
                continue
            if t.state == TrackState.TRACKED and t.track_id == 0 and t.hits >= 2:
                t.activate(self.frame_id)
            if t.state == TrackState.LOST:
                if t.frames_since_update <= self.cfg.max_lost:
                    next_lost.append(t)
                else:
                    t.state = TrackState.REMOVED
            elif t.state in (TrackState.TRACKED, TrackState.NEW):
                if t.frames_since_update == 0:
                    next_tracked.append(t)
                else:
                    t.mark_lost()
                    if t.state == TrackState.LOST:
                        next_lost.append(t)
                    # NEW tracks that missed immediately are dropped

        self.tracked = next_tracked
        self.lost = next_lost
        if self.cfg.cross_cut_reid:
            self._try_reid()
        if homography is not None:
            self._prev_H = homography
        return [t for t in self.tracked if t.track_id > 0]
