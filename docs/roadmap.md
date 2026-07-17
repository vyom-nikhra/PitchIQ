# Roadmap & known weaknesses

Prioritised improvements, informed by a survey of peer systems (Roboflow
`sports`, TrackNet family, `KumarranMahesh/datum`, and the 2025 broadcast-
tracking literature). Items 1–2 are **in progress**; 3–4 and the tracking
items are **documented weaknesses / planned work**.

## Landed

### 1. Embedding-based team assignment  ✅  *(fixed a real failure)*
Colour-histogram clustering collapsed on near-identical-tone kits (Real
Madrid white vs Man City sky-blue at 576p, ~13 vs 1 per frame). Now the
`embed` backend of `TeamAssigner` embeds player crops with a learned image
model — **torchvision CNN → UMAP → K-Means** (the Roboflow-`sports` recipe;
SigLIP auto-used when `transformers` is installed), tiered SigLIP → CNN →
colour fallback. Measured on the real clip: separability **1.7 → 3.1**,
per-frame split **~13:1 → ~8:5**, possession **97/3 → 62/38**. Enabled in
`configs/football.yaml`.

### 2. TrackNet-style ball tracker  ✅  *(trained on real footage)*
The ball is the least reliable component (small, fast, motion-blurred,
occluded); detection-based tracking (YOLO + Kalman + ROI) struggles. Now a
**TrackNet-style model regresses a ball heatmap from 3 consecutive frames**
(`perception/detection/tracknet.py`), integrated behind the ball interface
with fallback to the YOLO+Kalman selector. Synthetic domain: **97% / 1.6 px**.
Real domain, trained on SoccerNet-tracking ball GT (NDA-local) with the full
procedure in `scripts/train_ball_tracker.py` (lazy disk dataset, sequence-
level split, focal loss, bf16, augmentation, best-checkpoint/early-stop):
demo-clip ball coverage 37% → 95% of frames including airborne balls.
Head-to-head on sequences unseen by both models (SNMOT-060/070/099/109,
threshold sweep via `scripts/eval_ball_tracker.py`): the local bf16 model
gives **72% det / 27% fp @ 0.35, or 51% det / 8% fp @ 0.50, 0.8–0.9 px
median** — and dominates a Kaggle model trained on 2.2× the data but in
fp16 with overflow clamps (78%/56% @ 0.35). Lesson: training numerics beat
data volume here. Detection rate is honest-hard: the ball is genuinely
occluded/invisible in a large fraction of broadcast frames.
Next lifts, in expected order of value: trajectory post-processing over the
raw heatmap peaks (spline/Kalman with outlier rejection), hard-negative
mining of the false-positive frames, a 5-frame input window, higher input
resolution, and a full-length low-LR cosine tail (both runs early-stopped
before the schedule's fine-tuning phase).

### Confidence-aware reporting  ✅  *(2026-07-17 trust sprint)*
Perception quality (calibration coverage, observed-ball share, identity
stability, kit separability) is graded per clip from the cached artifacts
(`pitchiq/report/quality.py`) and propagated everywhere the system speaks:
the LLM report hedges ball-dependent claims when quality is low, the
dashboard shows a tracking-confidence badge, and a groundedness audit
(`scripts/audit_report.py`) verifies every numeric report claim against
`facts.json` (both demo reports: 100%).

## Planned (documented weaknesses)

### 3. Pose estimation for richer style embeddings
No per-player pose today. ViTPose/RTMPose joint keypoints would sharpen the
Layer-3 style embeddings (running gait, body orientation) and enable
action-level cues (shot vs pass vs tackle motion). `datum` uses ViTPose in
its CV stage for exactly this. *Medium effort; additive to `intelligence/`.*

### 4. Off-screen player imputation
Broadcast shows only part of the pitch, so at any moment ~half the outfield
players are off-frame. Formation detection, pitch control and Voronoi are
therefore computed on **whoever is visible**, which biases them (e.g. a
Voronoi tessellation missing 5 defenders overstates the attacking team's
space). A 2025 method (Royal Society Open Science 12:251175) estimates
continuous full-pitch positions from discrete broadcast data. Until then,
these metrics carry a visible-players caveat. *Medium effort; document
regardless.*

## Tracking robustness

- ✅ **Pitch-space max-speed association gate** (landed). The tracker now
  rejects any (track, detection) association whose implied real-pitch speed
  exceeds `tracking.max_assoc_speed_mps` (12.5 m/s), measured from the track's
  last *observed* foot point through the per-frame homography — so identity
  teleports are refused at association time, not just masked later by the
  analytics kinematics clamp. No-op on frames without a homography.

- ✅ **Cross-cut re-identification** (landed). On a scene cut every active
  identity is stashed (appearance embedding + last pitch position through the
  pre-cut homography); tracks born after the cut claim stashed IDs by
  appearance cosine distance, gated by an elapsed-time-aware pitch radius
  (`tracking.cross_cut_reid`, horizon `reid_horizon_s`). A player keeps their
  ID through a replay or camera change; strangers and right-shirt-wrong-place
  candidates are refused (unit-tested).

### Backlog
- Long same-kit occlusions in congestion (corners, goalmouths) still cause ID
  switches; jersey-number anchoring helps only when digits are legible.

## UI / UX backlog
- ✅ Downloadable report/CSV · richer radar (trails, hulls, pass arrows) ·
  first-visit tour · tracking-confidence badge — all landed.
- React frontend over the existing FastAPI (the app is already thin over
  `ArtifactStore` + JSON artifacts, so the port is mechanical).
