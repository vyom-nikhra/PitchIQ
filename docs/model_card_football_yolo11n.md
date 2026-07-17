---
license: agpl-3.0
tags:
  - object-detection
  - yolo
  - ultralytics
  - football
  - soccer
  - sports
library_name: ultralytics
---

# PitchIQ football detector — YOLOv11n fine-tune

Fine-tuned [Ultralytics YOLOv11n](https://github.com/ultralytics/ultralytics)
for broadcast football footage, with native classes for the four entities a
tactical-tracking pipeline needs. Trained for
[PitchIQ](https://github.com/vyom-nikhra/PitchIQ) — broadcast clip → metric
tracking table → tactical analytics → grounded LLM report.

| Class | mAP50 (held-out) |
|---|---|
| player | 0.993 |
| goalkeeper | 0.956 |
| referee | 0.977 |
| ball | 0.625 |

The ball number is honest, not a defect of this checkpoint: a tiny,
motion-blurred, frequently occluded object is the known-hard case for
single-frame box detectors. PitchIQ layers a Kalman-gated ROI second pass
(and optionally a TrackNet-style heatmap tracker) on top; if you use these
weights standalone, expect to need similar temporal help for the ball.

## Training

- Base: `yolo11n.pt` (COCO-pretrained), fine-tuned ~80 epochs at 1280 px on a
  Kaggle T4 via `scripts/train_detector.py` in the PitchIQ repo.
- Data: the open Roboflow Universe *football-players-detection* dataset
  (CC BY 4.0) — broadcast frames annotated with player / goalkeeper /
  referee / ball. **No SoccerNet or other restricted data was used.**

## Usage

```python
from ultralytics import YOLO

model = YOLO("football_yolo11n.pt")
results = model.predict("frame.jpg", conf=0.3, imgsz=1280)
```

Or run the full PitchIQ pipeline, which downloads this model automatically
(`detection.weights_url` in `configs/default.yaml`).

## Licence & attribution

- Weights are a derivative of Ultralytics YOLOv11 and are released under
  **AGPL-3.0** (the PitchIQ source code itself is MIT).
- Training data: Roboflow Universe football-players-detection (CC BY 4.0) —
  credit to the dataset authors.
