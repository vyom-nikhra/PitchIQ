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

### 2. TrackNet-style ball tracker  ✅  *(infrastructure + synthetic-validated)*
The ball is the least reliable component (small, fast, motion-blurred,
occluded); detection-based tracking (YOLO + Kalman + ROI) struggles. Now a
**TrackNet-style model regresses a ball heatmap from 3 consecutive frames**
(`perception/detection/tracknet.py`), integrated behind the ball interface
with fallback to the YOLO+Kalman selector. Validated end-to-end on synthetic
renders at **97% detection / 1.6 px median**. Training script
(`scripts/train_ball_tracker.py`) supports both the licence-clean synthetic
renderer and real SoccerNet-tracking ball GT — a real-broadcast ball tracker
needs the latter (NDA download), exactly like the keypoint model.

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

### Backlog
- **No cross-cut re-identification** *(next up)*. On a scene cut the tracker resets and
  assigns fresh IDs — each camera shot is tracked independently, with no
  identity carried across the cut. Proper continuity would match post-cut
  detections to pre-cut tracks via appearance embedding + last-known pitch
  position (a short-horizon re-ID), so a player keeps their ID through a
  replay or angle change.
- Long same-kit occlusions in congestion (corners, goalmouths) still cause ID
  switches; jersey-number anchoring helps only when digits are legible.

## UI / UX backlog
- Cleaner match selector, per-tab loading states, downloadable report/CSV.
- Richer radar (trails, team-shape hulls, pass arrows on the radar itself).
- A guided "tour" of a demo match for first-time visitors.
- React frontend over the existing FastAPI (the app is already thin over
  `ArtifactStore` + JSON artifacts, so the port is mechanical).
