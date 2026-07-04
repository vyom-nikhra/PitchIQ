"""Data download helper with the legal guardrails baked in.

    python scripts/download_data.py --metrica            # public (validation)
    python scripts/download_data.py --statsbomb          # public (validation)
    python scripts/download_data.py --roboflow           # open licence (training)
    python scripts/download_data.py --soccernet calibration   # NDA-RESTRICTED

SoccerNet downloads land in data/soccernet/ which is hard-ignored by git;
the NDA (non-commercial research, no redistribution) means that directory
must never be committed, uploaded, or shared — including to Kaggle.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SOCCERNET_WARNING = """
============================ NDA REMINDER =============================
SoccerNet data is KAUST Confidential Information under your NDA:
  * non-commercial research use only
  * NO redistribution: never commit, never upload (not even private Kaggle)
  * it stays in data/soccernet/ which is git-ignored — leave it there
=======================================================================
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrica", action="store_true")
    ap.add_argument("--statsbomb", action="store_true")
    ap.add_argument("--roboflow", action="store_true")
    ap.add_argument("--soccernet", choices=["calibration", "tracking", "jersey"],
                    default=None)
    args = ap.parse_args()

    from pitchiq.core.env import get_secret

    if args.metrica:
        dest = REPO / "data" / "downloads" / "metrica"
        if (dest / "data").exists():
            print(f"metrica already present at {dest}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/metrica-sports/sample-data.git",
                            str(dest)], check=True)
        print("Metrica sample data ready:", dest / "data")

    if args.statsbomb:
        from pitchiq.io.statsbomb import list_open_matches, load_events

        matches = list_open_matches()
        print(f"cached competition index ({len(matches)} matches). Example:")
        print(matches.head(3).to_string(index=False))
        mid = int(matches.iloc[0].match_id)
        ev = load_events(mid)
        print(f"cached events for match {mid}: {len(ev)} events")

    if args.roboflow:
        from pitchiq.io.roboflow import download_dataset

        path = download_dataset()
        print("Roboflow dataset ready:", path)

    if args.soccernet:
        print(SOCCERNET_WARNING)
        password = get_secret("SOCCERNET_PASSWORD")
        if not password:
            raise SystemExit("SOCCERNET_PASSWORD missing from .env")
        from pitchiq.io.soccernet import download_soccernet

        dest = REPO / "data" / "soccernet" / args.soccernet
        dest.mkdir(parents=True, exist_ok=True)
        download_soccernet(dest, task=args.soccernet, password=password)
        print("done:", dest)
        print(SOCCERNET_WARNING)


if __name__ == "__main__":
    main()
