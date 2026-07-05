"""Standalone Kaggle kernel: fine-tune YOLOv11 on the Roboflow football set.

Self-contained on purpose (no pitchiq import) so it can be pasted into a
Kaggle *script* kernel as-is. Dataset licence (CC BY 4.0) permits Kaggle use —
unlike SoccerNet, which must never be uploaded there.

Setup on Kaggle:
  1. New Notebook → Accelerator: GPU (T4/P100) → paste this file.
  2. Add-ons → Secrets → add ROBOFLOW_API_KEY (free at roboflow.com).
  3. Run. ~40–60 min on a T4 for 80 epochs.
  4. Download /kaggle/working/runs/detector/football/weights/best.pt and set
     `detection.weights` to it in your PitchIQ config (see configs/football.yaml).
"""

import os
import subprocess
import sys


def sh(cmd: str) -> None:
    print("+", cmd)
    subprocess.run(cmd, shell=True, check=True)


sh(f"{sys.executable} -m pip install -q ultralytics roboflow")

# Roboflow key from Kaggle secrets (falls back to env for local runs)
try:
    from kaggle_secrets import UserSecretsClient

    os.environ["ROBOFLOW_API_KEY"] = UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
except Exception:
    pass
assert os.environ.get("ROBOFLOW_API_KEY"), "set ROBOFLOW_API_KEY (Kaggle secret)"

from roboflow import Roboflow  # noqa: E402

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
dataset = (rf.workspace("roboflow-jvuqo")
             .project("football-players-detection-3zvbc")
             .version(12)
             .download("yolov11"))
print("dataset:", dataset.location)

from ultralytics import YOLO  # noqa: E402

model = YOLO("yolo11n.pt")  # nano: best speed/VRAM tradeoff for PitchIQ CPU/edge use
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=80,
    imgsz=1280,
    batch=-1,          # auto-fit VRAM
    project="runs/detector",
    name="football",
    patience=20,
)
metrics = model.val()
print({"mAP50-95": float(metrics.box.map), "mAP50": float(metrics.box.map50)})
print("weights: runs/detector/football/weights/best.pt")
