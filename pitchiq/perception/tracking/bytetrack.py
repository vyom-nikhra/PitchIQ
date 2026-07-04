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
        return cost.astype(np.float32)

    def update(
        self, detections: list[Detection], camera_affine: np.ndarray | None = None
    ) -> list[STrack]:
        """Advance one frame; returns currently tracked (active) tracks."""
        self.frame_id += 1
        dets = [d for d in detections if d.cls != EntityClass.BALL]
        high = [d for d in dets if d.conf >= self.cfg.high_thresh]
        low = [d for d in dets if self.cfg.low_thresh <= d.conf < self.cfg.high_thresh]

        for t in self.tracked + self.lost:
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
        return [t for t in self.tracked if t.track_id > 0]
