"""Per-match artifact store: one directory per processed match/job.

Layout::

    <job_dir>/
        input.mp4                 # source clip (optional for pure-data matches)
        meta.json                 # MatchMeta
        status.json               # stage/progress for the app
        tracking.parquet          # THE tracking table
        homography.parquet
        events.parquet            # possession spells / passes / turnovers
        analytics/*.json|*.npz
        intelligence/*.json|*.parquet|*.npz
        report/report.md, facts.json
        media/annotated.mp4, radar.html

The store is deliberately dumb (paths + json helpers): every producer writes
through it so the Streamlit app and FastAPI serve from one canonical layout.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pitchiq.core.schema import (
    MatchMeta,
    load_homographies,
    load_tracking_table,
    save_homographies,
    save_tracking_table,
)


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- paths
    @property
    def input_video(self) -> Path:
        return self.root / "input.mp4"

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def tracking_path(self) -> Path:
        return self.root / "tracking.parquet"

    @property
    def homography_path(self) -> Path:
        return self.root / "homography.parquet"

    @property
    def events_path(self) -> Path:
        return self.root / "events.parquet"

    def analytics_path(self, name: str) -> Path:
        p = self.root / "analytics" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def intelligence_path(self, name: str) -> Path:
        p = self.root / "intelligence" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def report_path(self, name: str) -> Path:
        p = self.root / "report" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def media_path(self, name: str) -> Path:
        p = self.root / "media" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------ core IO
    def save_meta(self, meta: MatchMeta) -> None:
        meta.save(self.meta_path)

    def load_meta(self) -> MatchMeta:
        return MatchMeta.load(self.meta_path)

    def save_tracking(self, df: pd.DataFrame) -> None:
        save_tracking_table(df, self.tracking_path)

    def load_tracking(self) -> pd.DataFrame:
        return load_tracking_table(self.tracking_path)

    def save_homography(self, df: pd.DataFrame) -> None:
        save_homographies(df, self.homography_path)

    def load_homography(self) -> pd.DataFrame:
        return load_homographies(self.homography_path)

    def save_events(self, df: pd.DataFrame) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.events_path, index=False)

    def load_events(self) -> pd.DataFrame:
        return pd.read_parquet(self.events_path)

    def has_tracking(self) -> bool:
        return self.tracking_path.exists() and self.meta_path.exists()

    # ------------------------------------------------------------ json/npz
    def save_json(self, path: Path, obj: dict | list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")

    def load_json(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def save_npz(self, path: Path, **arrays: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    # ------------------------------------------------------------- status
    def update_status(self, stage: str, progress: float, message: str = "", state: str = "running") -> None:
        """Progress heartbeat consumed by the web app. ``progress`` in [0, 1]."""
        payload = {
            "state": state,  # queued | running | done | error
            "stage": stage,
            "progress": round(float(progress), 4),
            "message": message,
            "updated_at": time.time(),
        }
        (self.root / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    def read_status(self) -> dict:
        p = self.root / "status.json"
        if not p.exists():
            return {"state": "unknown", "stage": "", "progress": 0.0, "message": ""}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"state": "unknown", "stage": "", "progress": 0.0, "message": "unreadable status"}


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")
