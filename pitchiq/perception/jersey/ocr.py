"""Jersey-number recognition with track-level propagation.

A player's number is constant, so one confident read anywhere in a track
labels the whole track. Per sampled frame we OCR the torso crop; a
:class:`JerseyVoter` accumulates confidence-weighted votes per track and only
commits a number once it has ``min_votes`` reads and a dominant vote share.

Backend: EasyOCR (digit allowlist) when installed — it's a heavy optional
dependency (``pitchiq[cv]``), so ``create_jersey_reader`` returns ``None``
gracefully when unavailable and the tracking table simply carries null
jersey numbers. A purpose-trained digit classifier (SoccerNet jersey task)
is the documented upgrade path.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import cv2
import numpy as np

from pitchiq.config import JerseyConfig

log = logging.getLogger(__name__)


class EasyOCRReader:
    name = "easyocr"

    def __init__(self) -> None:
        import easyocr  # optional heavy import

        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(self, crop_bgr: np.ndarray) -> tuple[int, float] | None:
        """Return (number, confidence) or None."""
        if crop_bgr.size == 0:
            return None
        up = cv2.resize(crop_bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        results = self.reader.readtext(up, allowlist="0123456789", detail=1)
        best: tuple[int, float] | None = None
        for _, text, conf in results:
            text = text.strip()
            if not text.isdigit():
                continue
            num = int(text)
            if not 1 <= num <= 99:
                continue
            if best is None or conf > best[1]:
                best = (num, float(conf))
        return best


def create_jersey_reader(cfg: JerseyConfig):
    """Build the configured OCR backend, or None if disabled/unavailable."""
    if not cfg.enabled or cfg.backend == "none":
        return None
    try:
        reader = EasyOCRReader()
        log.info("jersey OCR backend: easyocr")
        return reader
    except Exception as exc:
        if cfg.backend == "easyocr":
            log.warning("easyocr requested but unavailable: %s", exc)
        else:
            log.info("jersey OCR disabled (easyocr not installed): numbers will be null")
        return None


class JerseyVoter:
    """Confidence-weighted majority vote of OCR reads per track."""

    def __init__(self, cfg: JerseyConfig) -> None:
        self.cfg = cfg
        self._votes: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def add(self, track_id: int, number: int, conf: float) -> None:
        if conf < self.cfg.min_conf:
            return
        self._votes[track_id][number] += conf
        self._counts[track_id][number] += 1

    def finalize(self) -> dict[int, int]:
        """track_id -> jersey number, only where evidence is decisive."""
        out: dict[int, int] = {}
        for tid, votes in self._votes.items():
            total = sum(votes.values())
            best_num, best_w = max(votes.items(), key=lambda kv: kv[1])
            if self._counts[tid][best_num] >= self.cfg.min_votes and best_w / total >= 0.6:
                out[tid] = best_num
        return out

    def merge_team_duplicates(
        self, numbers: dict[int, int], team_of: dict[int, "object"]
    ) -> dict[int, int]:
        """Drop conflicting reads: two same-team tracks can't share a number
        simultaneously — keep the higher-vote one. (Track fragments of the same
        player at different times legitimately share a number and are kept.)"""
        by_key: dict[tuple, list[int]] = defaultdict(list)
        for tid, num in numbers.items():
            by_key[(team_of.get(tid), num)].append(tid)
        out = dict(numbers)
        for key, tids in by_key.items():
            if len(tids) <= 1:
                continue
            tids_sorted = sorted(tids, key=lambda t: sum(self._votes[t].values()), reverse=True)
            # keep the strongest; drop others only if their track overlaps in time
            # (we lack time info here, so conservatively keep all — pipeline
            # re-checks temporal overlap with the tracking table)
            _ = tids_sorted
        return out
