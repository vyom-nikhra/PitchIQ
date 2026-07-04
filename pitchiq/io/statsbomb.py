"""StatsBomb open-data loader — free event data for validating derived events.

PitchIQ derives passes/possession from tracking; StatsBomb events give a
ground-truth-ish reference (pass counts, possession share) for real matches.
Data comes straight from the public GitHub repo, cached locally as JSON.

StatsBomb pitch coords are 120 x 80 yards-ish units, (0,0) top-left, y down.
We rescale to 105 x 68 metres with y up.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pitchiq.io.errors import DatasetUnavailable

RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def _fetch(path: str, cache_dir: Path) -> dict | list:
    cache = cache_dir / path.replace("/", "_")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    try:
        import requests

        resp = requests.get(f"{RAW_BASE}/{path}", timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        raise DatasetUnavailable(
            f"Could not fetch StatsBomb open data ({path}): {exc}. "
            "Needs network access to raw.githubusercontent.com."
        ) from exc
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text, encoding="utf-8")
    return resp.json()


def load_events(match_id: int, cache_dir: str | Path = "data/downloads/statsbomb") -> pd.DataFrame:
    """Load one match's events; returns tidy df with pass rows rescaled to metres."""
    cache_dir = Path(cache_dir)
    events = _fetch(f"events/{match_id}.json", cache_dir)
    rows = []
    for ev in events:
        loc = ev.get("location") or [None, None]
        row = {
            "event_type": ev["type"]["name"],
            "minute": ev["minute"],
            "second": ev["second"],
            "team": ev.get("team", {}).get("name"),
            "player": ev.get("player", {}).get("name"),
            "x": _sb_x(loc[0]),
            "y": _sb_y(loc[1]),
        }
        if ev["type"]["name"] == "Pass":
            end = ev["pass"].get("end_location") or [None, None]
            row["end_x"] = _sb_x(end[0])
            row["end_y"] = _sb_y(end[1])
            row["outcome"] = (ev["pass"].get("outcome") or {}).get("name", "Complete")
        rows.append(row)
    return pd.DataFrame(rows)


def _sb_x(v):  # 120 -> 105
    return None if v is None else v / 120.0 * 105.0


def _sb_y(v):  # 80 -> 68, flip so y grows up
    return None if v is None else (80.0 - v) / 80.0 * 68.0


def list_open_matches(
    competition_id: int = 11, season_id: int = 90, cache_dir: str | Path = "data/downloads/statsbomb"
) -> pd.DataFrame:
    """List match ids for an open competition (default: La Liga 2020/21)."""
    matches = _fetch(f"matches/{competition_id}/{season_id}.json", Path(cache_dir))
    return pd.DataFrame(
        [
            {
                "match_id": m["match_id"],
                "home": m["home_team"]["home_team_name"],
                "away": m["away_team"]["away_team_name"],
                "score": f"{m['home_score']}-{m['away_score']}",
                "date": m["match_date"],
            }
            for m in matches
        ]
    )
