"""Fine-tune YOLOv11 (or RT-DETR) on the Roboflow football detection dataset.

Produces football-native classes (ball/goalkeeper/player/referee), removing
the COCO-fallback limitation. Dataset: 'football-players-detection' on
Roboflow Universe (open licence) — safe to train anywhere, including Kaggle.

Local (RTX 3050 ~1-2 h):
    set ROBOFLOW_API_KEY in .env, then
    python scripts/train_detector.py --model yolo11n.pt --epochs 80

Kaggle (free GPU): create a notebook, `pip install ultralytics roboflow`,
add ROBOFLOW_API_KEY as a Kaggle secret, paste this file, run main().
Afterwards set `detection.weights` in your config to the produced best.pt.

--benchmark runs the YOLO-vs-RT-DETR comparison on the validation split.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt",
                    help="base weights: yolo11n/s/m.pt or rtdetr-l.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto for VRAM")
    ap.add_argument("--data-dir", default="data/downloads/roboflow")
    ap.add_argument("--benchmark", action="store_true",
                    help="compare YOLOv11 vs RT-DETR speed/accuracy on val")
    args = ap.parse_args()

    from pitchiq.core.env import load_env
    from pitchiq.io.roboflow import download_dataset

    load_env()
    ds_root = download_dataset(out_dir=args.data_dir)
    data_yaml = ds_root / "data.yaml"
    print(f"dataset ready: {data_yaml}")

    from ultralytics import RTDETR, YOLO

    def make(model_path: str):
        return RTDETR(model_path) if "rtdetr" in model_path else YOLO(model_path)

    if args.benchmark:
        results = {}
        for m in ("yolo11n.pt", "rtdetr-l.pt"):
            print(f"=== benchmarking {m} (short fine-tune) ===")
            model = make(m)
            model.train(data=str(data_yaml), epochs=max(10, args.epochs // 4),
                        imgsz=args.imgsz, batch=args.batch, project="runs/benchmark",
                        name=m.replace(".pt", ""))
            metrics = model.val(data=str(data_yaml))
            t0 = time.time()
            model.predict([str(p) for p in (ds_root / "valid" / "images").glob("*.jpg")][:50],
                          imgsz=args.imgsz, verbose=False)
            dt = (time.time() - t0) / 50
            results[m] = {"mAP50-95": float(metrics.box.map),
                          "mAP50": float(metrics.box.map50),
                          "s_per_image": round(dt, 3)}
        print("\nbenchmark:", results)
        return

    model = make(args.model)
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project="runs/detector", name="football")
    print("\nDone. Point `detection.weights` at "
          "runs/detector/football/weights/best.pt")


if __name__ == "__main__":
    main()
