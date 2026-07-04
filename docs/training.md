# Training guide

Three trainable components, in ascending order of hardware needs. All
training is optional — every consumer has a documented fallback.

## 1. Style encoder (contrastive, 6.1b) — CPU, minutes

```bash
python scripts/train_style_encoder.py --matches 8 --epochs 60
```

Trains on simulated matches (licence-clean); positive pairs are the same
player's phase-conditioned heatmaps from different halves. Output
`weights/style_encoder.pt` is picked up when `embeddings.learned.weights`
points at it (the demo build does this automatically). To include real
matches, process them first, then extend the pair builder with per-half
features from `data/jobs/*`.

## 2. Detector fine-tune (YOLOv11 / RT-DETR) — GPU, ~1–2 h

Dataset: Roboflow `football-players-detection` (open licence). Needs
`ROBOFLOW_API_KEY` in `.env`.

Local (RTX 3050-class works: yolo11n @ 1280, auto batch):

```bash
python scripts/train_detector.py --model yolo11n.pt --epochs 80
# YOLO vs RT-DETR comparison study:
python scripts/train_detector.py --benchmark
```

Kaggle (free 30 h/week GPU): new notebook → `pip install ultralytics
roboflow` → add `ROBOFLOW_API_KEY` as a Kaggle secret → paste
`scripts/train_detector.py` → run. Download `best.pt` and set
`detection.weights` in your config. This dataset's licence permits Kaggle
use — **SoccerNet's does not; never upload SoccerNet data anywhere.**

Effect: native ball/goalkeeper/player/referee classes (removes the COCO
fallback heuristics) and a much better ball detector.

## 3. Pitch keypoint model — GPU, local only (NDA data)

```bash
python scripts/download_data.py --soccernet calibration   # ~3 GB, NDA
python scripts/train_pitch_keypoints.py --epochs 30
```

Trains the heatmap U-Net on keypoints derived from SoccerNet-Calibration
line annotations. Set `calibration.keypoint_weights:
weights/pitch_keypoints.pt` — the calibrator then prefers keypoints and
falls back to lines/conics automatically. This is the fix for box-only
views. Weights derived from NDA data stay out of the public repo by default
(see `docs/data_sources.md`).

## Experiment tracking

All ultralytics runs log locally under `runs/`; set `WANDB_API_KEY` and pass
`--project` to route them to Weights & Biases if desired (optional extra
`pitchiq[track]`).
