# Roadmap & known weaknesses

Prioritised improvements, informed by a survey of peer systems (Roboflow
`sports`, TrackNet family, `KumarranMahesh/datum`, and the 2025 broadcast-
tracking literature). Items 1–2 are **in progress**; 3–4 and the tracking
items are **documented weaknesses / planned work**.

## In progress

### 1. Embedding-based team assignment  *(fixes a real failure)*
Colour-histogram clustering collapses on near-identical-tone kits at low
resolution (measured: Real Madrid white vs Man City sky-blue at 576p lumped
~13 vs 1 per frame). Peer systems (Roboflow `sports`) solve this with a
learned image embedding — **SigLIP crop embeddings → UMAP → K-Means** — which
separates teams that colour statistics can't. Implemented as the `embed`
backend of `TeamAssigner`, tiered SigLIP → torchvision-CNN → colour fallback.

### 2. TrackNet-style ball tracker  *(fixes our weakest link)*
The ball is the least reliable component (small, fast, motion-blurred,
occluded). Detection-based tracking (YOLO + Kalman + ROI) fundamentally
struggles here. The field standard is **TrackNet** (Huang et al. 2019) and
successors (V4 motion-attention, TOTNet occlusion-aware): regress a ball
**probability heatmap from 3+ consecutive frames**, learning the trajectory
pattern instead of detecting a box. Reported ~97% recall on exactly the
blurry/tiny/afterimage cases that break box detectors.

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

## Tracking robustness backlog

- **No pitch-space max-speed association gate.** The tracker associates in
  pixel space (IoU + Kalman constant-velocity), which soft-discourages but
  does not forbid physically impossible jumps; only the analytics kinematics
  layer hard-clamps speeds > 11 m/s. Adding a metre-space velocity gate
  (reject any association implying > ~11 m/s of real-pitch motion, using the
  per-frame homography) would reject identity teleports at the source.
- **No cross-cut re-identification.** On a scene cut the tracker resets and
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
