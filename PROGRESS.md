# PROGRESS

Running status of every PitchIQ component. Updated as the build proceeds.

Legend: ✅ fully functional · 🟡 functional with documented fallback/limitation · 🔧 stubbed interface · ⬜ not started

## Phase 0 — Foundations
- ⬜ Project scaffolding, config system
- ⬜ Core domain models (pitch, schema, types)
- ⬜ Data loaders (SoccerNet / Roboflow / StatsBomb / Metrica)
- ⬜ Detection (YOLOv11 / RT-DETR / fallback)
- ⬜ Tracking (ByteTrack + re-ID)
- ⬜ Team assignment
- ⬜ Jersey OCR

## Phase 1 — Homography + tactical map
- ⬜ Pitch template & keypoints
- ⬜ Line-based calibration (DLT + RANSAC)
- ⬜ Keypoint-model calibration (optional path)
- ⬜ Temporal smoothing / scene cuts / flow propagation
- ⬜ Tracking table (parquet) + caching
- ⬜ Tactical radar synced to video

## Phase 2 — Core analytics
- ⬜ Kinematics (speed / distance / sprints)
- ⬜ Possession
- ⬜ Heatmaps
- ⬜ Formations + convex hull spread
- ⬜ Voronoi / pitch control
- ⬜ Defensive line height & compactness
- ⬜ Territory / field tilt

## Phase 3 — Intelligence
- ⬜ Player-style features (handcrafted)
- ⬜ Learned embedding (contrastive)
- ⬜ Role discovery
- ⬜ Similar-player search
- ⬜ Marking analysis

## Phase 4 — Advanced analytics + report
- ⬜ Pass detection / pass networks
- ⬜ Line-breaking passes
- ⬜ Expected Threat (xT)
- ⬜ Pressing / PPDA
- ⬜ Formation transitions, phase segmentation
- ⬜ LLM report + Q&A

## Phase 5 — Web app + deployment
- ⬜ FastAPI backend, async jobs
- ⬜ Streamlit dashboard (5 tabs)
- ⬜ Bundled demo match
- ⬜ Docker
- ⬜ README / docs

## Phase 6 — Stretch
- ⬜ Team-style fingerprint
- ⬜ Off-ball run valuation
- ⬜ RT-DETR vs YOLO benchmark harness
