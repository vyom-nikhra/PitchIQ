"""SoccerNet loader/downloader wiring.

SoccerNet provides (with a free NDA password for videos):
  * tracking (SoccerNet-Tracking) — MOT-format player/ball tracks,
  * calibration (SoccerNet-Calibration) — pitch-line/keypoint annotations,
  * jersey numbers, re-ID crops, action spotting.

We wire the official ``SoccerNet`` pip package for downloads and provide
parsers for the two products PitchIQ consumes: MOT tracking ground truth (to
score our tracker: MOTA/IDF1) and calibration annotations (to train/evaluate
the pitch-keypoint model). Videos are NOT bundled; without the NDA password
the loader raises :class:`DatasetUnavailable` with instructions.
"""

from __future__ import annotations

import configparser
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from pitchiq.io.errors import DatasetUnavailable

SOCCERNET_SITE = "https://www.soccer-net.org/"


def download_soccernet(local_dir: str | Path, task: str = "tracking", password: str | None = None) -> Path:
    """Download a SoccerNet task locally (requires the ``SoccerNet`` package).

    ``task``: 'tracking' | 'calibration' | 'jersey' | 'reid'.
    """
    local_dir = Path(local_dir)
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError as exc:
        raise DatasetUnavailable(
            "pip install SoccerNet  (and register at "
            f"{SOCCERNET_SITE} for the video NDA password)"
        ) from exc
    dl = SoccerNetDownloader(LocalDirectory=str(local_dir))
    if password:
        dl.password = password
    if task == "tracking":
        dl.downloadDataTask(task="tracking", split=["train", "test"])
    elif task == "calibration":
        dl.downloadDataTask(task="calibration", split=["train", "valid", "test"])
    elif task == "jersey":
        dl.downloadDataTask(task="jersey-2023", split=["train", "test"])
    elif task == "reid":
        dl.downloadDataTask(task="reid", split=["train", "valid", "test"])
    else:
        raise ValueError(f"unknown task {task}")
    return local_dir


def load_mot_ground_truth(sequence_dir: str | Path) -> pd.DataFrame:
    """Parse one SoccerNet-Tracking MOT sequence (``gt/gt.txt`` + ``seqinfo.ini``).

    Returns a dataframe with columns
    ``frame, track_id, x1, y1, x2, y2, conf`` in pixels — the format
    :mod:`pitchiq.perception.tracking.metrics` scores against.
    """
    sequence_dir = Path(sequence_dir)
    gt_path = sequence_dir / "gt" / "gt.txt"
    if not gt_path.exists():
        raise DatasetUnavailable(
            f"No MOT ground truth at {gt_path}. Download SoccerNet tracking via "
            "pitchiq.io.soccernet.download_soccernet(local_dir, task='tracking')."
        )
    cols = ["frame", "track_id", "x", "y", "w", "h", "conf", "a", "b", "c"]
    gt = pd.read_csv(gt_path, header=None, names=cols[: len(pd.read_csv(gt_path, header=None, nrows=1).columns)])
    out = pd.DataFrame(
        {
            "frame": gt["frame"].astype(int) - 1,  # MOT is 1-based
            "track_id": gt["track_id"].astype(int),
            "x1": gt["x"].astype(float),
            "y1": gt["y"].astype(float),
            "x2": (gt["x"] + gt["w"]).astype(float),
            "y2": (gt["y"] + gt["h"]).astype(float),
            "conf": gt.get("conf", pd.Series(np.ones(len(gt)))).astype(float),
        }
    )
    info = sequence_dir / "seqinfo.ini"
    if info.exists():
        cp = configparser.ConfigParser()
        cp.read(info)
        out.attrs["fps"] = float(cp.get("Sequence", "frameRate", fallback="25"))
    return out


def iter_calibration_samples(calib_dir: str | Path, split: str = "train"):
    """Yield ``(image_path, lines_dict)`` from SoccerNet-Calibration.

    ``lines_dict`` maps semantic line names (e.g. 'Big rect. left main') to
    lists of normalised image points — the supervision used by
    ``scripts/train_pitch_keypoints.py``.
    """
    import json

    calib_dir = Path(calib_dir)
    split_dir = calib_dir / split
    zip_path = calib_dir / f"{split}.zip"
    if not split_dir.exists() and zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(calib_dir)
    if not split_dir.exists():
        raise DatasetUnavailable(
            f"SoccerNet-Calibration split '{split}' not found under {calib_dir}. "
            "Download via download_soccernet(local_dir, task='calibration')."
        )
    for img in sorted(split_dir.glob("*.jpg")):
        ann = img.with_suffix(".json")
        if ann.exists():
            yield img, json.loads(ann.read_text(encoding="utf-8"))
