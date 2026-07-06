"""Reconstruct demo clips from SoccerNet-Tracking sequences.

SoccerNet-Tracking sequences are short (~30 s), main-camera, tracking-friendly
clips stored as ``img1/*.jpg`` frames — ideal for a video demonstration. This
reassembles selected sequences into mp4 clips under ``data/raw/`` (git-ignored)
and prints a one-line kit/colour summary so a sequence with visually distinct
kits (which shows the team-assignment upgrade at its best) can be chosen.

NDA: SoccerNet content is local-only — never commit the produced clips.

Usage:
    python scripts/extract_soccernet_clip.py --list          # summarise sequences
    python scripts/extract_soccernet_clip.py --seq SNMOT-116 # build one clip
    python scripts/extract_soccernet_clip.py --best 3        # build the 3 most
                                                             #   colour-distinct
"""

from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _seq_dirs(root: Path) -> list[Path]:
    seqs = [d for d in sorted(root.rglob("*")) if (d / "img1").is_dir()]
    if seqs:
        return seqs
    # not extracted yet — unzip any SoccerNet tracking zips in place
    import zipfile

    for zp in root.rglob("*.zip"):
        print(f"extracting {zp.name} ...", flush=True)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(zp.parent)
    return [d for d in sorted(root.rglob("*")) if (d / "img1").is_dir()]


def _kit_distinctness(seq: Path, n_probe: int = 8) -> tuple[float, list[str]]:
    """Rough two-kit colour separation: cluster player-ish crops from a few
    frames by mean colour and return inter-cluster distance + the two hex means.
    Higher = more visually distinct kits = better demo of team assignment."""
    imgs = sorted((seq / "img1").glob("*.jpg"))
    if len(imgs) < n_probe:
        return 0.0, []
    picks = imgs[:: max(1, len(imgs) // n_probe)][:n_probe]
    means = []
    for ip in picks:
        fr = cv2.imread(str(ip))
        if fr is None:
            continue
        h, w = fr.shape[:2]
        # sample small central-lower windows (torsos scattered on the pitch)
        for _ in range(12):
            x = np.random.randint(0, w - 20)
            y = np.random.randint(int(h * 0.2), h - 40)
            patch = fr[y:y + 30, x:x + 15]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            if (hsv[:, :, 1] > 60).mean() > 0.3:  # saturated (kit, not grass/white)
                means.append(patch.reshape(-1, 3).mean(0))
    if len(means) < 6:
        return 0.0, []
    from sklearn.cluster import KMeans

    X = np.stack(means)
    km = KMeans(2, n_init=5, random_state=0).fit(X)
    c0, c1 = km.cluster_centers_
    dist = float(np.linalg.norm(c0 - c1))
    hexes = [f"#{int(c[2]):02x}{int(c[1]):02x}{int(c[0]):02x}" for c in (c0, c1)]
    return dist, hexes


def build_clip(seq: Path, out_dir: Path, max_seconds: float = 60.0) -> Path:
    info = configparser.ConfigParser()
    seqinfo = seq / "seqinfo.ini"
    fps = 25.0
    if seqinfo.exists():
        info.read(seqinfo)
        fps = float(info.get("Sequence", "frameRate", fallback="25"))
    imgs = sorted((seq / "img1").glob("*.jpg"))
    imgs = imgs[: int(max_seconds * fps)]
    first = cv2.imread(str(imgs[0]))
    h, w = first.shape[:2]
    from pitchiq.core.video import VideoSink

    out = out_dir / f"soccernet_{seq.name}.mp4"
    with VideoSink(out, fps, (w, h)) as sink:
        for ip in imgs:
            fr = cv2.imread(str(ip))
            if fr is not None:
                sink.write(fr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/soccernet/tracking")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seq", default=None)
    ap.add_argument("--best", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    args = ap.parse_args()

    root = REPO / args.data_dir
    seqs = _seq_dirs(root)
    if not seqs:
        raise SystemExit(f"no tracking sequences under {root} (still downloading?)")
    out_dir = REPO / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seq:
        seq = next((s for s in seqs if s.name == args.seq), None)
        if seq is None:
            raise SystemExit(f"sequence {args.seq} not found")
        print("built:", build_clip(seq, out_dir, args.max_seconds))
        return

    print(f"{len(seqs)} sequences. Scoring kit distinctness (higher = better demo)...")
    scored = []
    for s in seqs:
        d, hexes = _kit_distinctness(s)
        scored.append((d, s, hexes))
        print(f"  {s.name:16s} kit-distinctness {d:6.1f}  kits {hexes}")
    scored.sort(key=lambda t: -t[0])

    if args.best:
        for d, s, hexes in scored[: args.best]:
            print(f"building {s.name} (distinctness {d:.1f})...")
            print("  ->", build_clip(s, out_dir, args.max_seconds))


if __name__ == "__main__":
    main()
