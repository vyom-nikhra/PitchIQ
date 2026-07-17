"""Publish the fine-tuned football detector to a public HF model repo.

Uploads weights/football_yolo11n.pt + the model card so the app's
detection.weights_url auto-download works for the public Space and fresh
clones. LEGAL: this model was trained ONLY on the open Roboflow dataset
(CC BY 4.0) — it is NOT NDA-bound. Never point this script at
pitch_keypoints.pt or ball_tracknet_real.pt (SoccerNet NDA, local-only).

Usage:  python scripts/publish_detector.py [--repo NuclearPanda/pitchiq-football-yolo11n]
Requires HF_TOKEN in .env (write access). The token is never printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pitchiq.core.env import get_secret  # noqa: E402

WEIGHTS = REPO / "weights" / "football_yolo11n.pt"
CARD = REPO / "docs" / "model_card_football_yolo11n.md"
NDA_NAMES = {"pitch_keypoints", "ball_tracknet_real", "ball_tracknet_kaggle",
             "ball_tracknet"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="NuclearPanda/pitchiq-football-yolo11n")
    args = ap.parse_args()

    if any(n in WEIGHTS.stem for n in NDA_NAMES):
        raise SystemExit("refusing: NDA-bound weights must never be uploaded")
    if not WEIGHTS.exists():
        raise SystemExit(f"weights not found: {WEIGHTS}")
    if not CARD.exists():
        raise SystemExit(f"model card not found: {CARD}")
    token = get_secret("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN missing from environment/.env")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="model")
    api.upload_file(path_or_fileobj=str(WEIGHTS),
                    path_in_repo="football_yolo11n.pt",
                    repo_id=args.repo, repo_type="model")
    print(f"published: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
