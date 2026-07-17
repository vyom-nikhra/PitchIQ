"""Roboflow football-detection dataset wiring (for YOLO fine-tuning).

The 'football-players-detection' family of datasets on Roboflow Universe is
pre-annotated with exactly PitchIQ's classes: ball / goalkeeper / player /
referee. ``download_dataset`` pulls it in YOLOv11 format ready for
``scripts/train_detector.py``. Requires a free ROBOFLOW_API_KEY.
"""

from __future__ import annotations

from pathlib import Path

from pitchiq.io.errors import DatasetUnavailable

DEFAULT_WORKSPACE = "roboflow-jvuqo"
DEFAULT_PROJECT = "football-players-detection-3zvbc"
DEFAULT_VERSION = 12

#: class order in the Roboflow dataset -> PitchIQ entity classes
ROBOFLOW_CLASS_MAP = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}


def download_dataset(
    out_dir: str | Path = "data/downloads/roboflow",
    workspace: str = DEFAULT_WORKSPACE,
    project: str = DEFAULT_PROJECT,
    version: int = DEFAULT_VERSION,
    fmt: str = "yolov11",
) -> Path:
    """Download the annotated football detection dataset in YOLO format."""
    from pitchiq.core.env import get_secret

    api_key = get_secret("ROBOFLOW_API_KEY")
    if not api_key:
        raise DatasetUnavailable(
            "Set ROBOFLOW_API_KEY (free at https://roboflow.com) to download the "
            f"'{project}' dataset for detector fine-tuning."
        )
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise DatasetUnavailable("pip install roboflow") from exc
    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download(
        fmt, location=str(Path(out_dir) / f"{project}-v{version}")
    )
    return Path(ds.location)
