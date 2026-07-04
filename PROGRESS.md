# PROGRESS

Running status of every PitchIQ component. Updated as the build proceeds.

Legend: ✅ fully functional · 🟡 functional with documented fallback/limitation · 🔧 stubbed interface · ⬜ not started

## Phase 0 — Foundations
- ✅ Project scaffolding, config system (pydantic + YAML deep-merge, config hash for cache invalidation)
- ✅ Core domain models (pitch w/ 33 keypoints, tracking-table schema, parquet IO, MatchMeta, ArtifactStore)
- 🟡 Data loaders — Metrica tracking + StatsBomb events + SoccerNet (tracking/calibration) + Roboflow parsers wired; actual downloads need network/API keys/NDA password and are not bundled
- 🟡 Detection — YOLOv11 & RT-DETR behind one interface (ultralytics installed, CPU); no football fine-tune bundled → COCO fallback (person/sports-ball) documented; colour-blob fallback detector works on synthetic renders with zero torch dependency
- ✅ Tracking — self-contained ByteTrack (Kalman xyah + two-stage Hungarian association), appearance blending (colour-hist embedder; OSNet torchscript hook), camera-motion compensation via background optical flow
- ✅ Team assignment — grass-suppressed LAB+hue kit signatures, K-Means(2), outlier→referee/GK heuristics, GK team via defended side, kit hex colours for viz
- 🟡 Jersey OCR — easyocr backend + confidence-weighted track voting; easyocr not installed in dev env → numbers null (documented)
- ✅ Tracking metrics — self-implemented MOTA / IDF1 / ID-switches

## Phase 1 — Homography + tactical map (the crux)
- ✅ Pitch template & keypoints (FIFA-standard, parametric)
- ✅ Line extraction — grass mask → top-hat white mask → Hough → collinear merge → orientation families
- ✅ Line-based calibration — order-preserving template assignment search, DLT + RANSAC, mask-coverage scoring with plausibility gates
- ✅ Conic calibration — ellipse RANSAC over arc components (contiguity-safe), projectively exact constructions: circle∩halfway, tangent points from halfway×touchline corner (breaks the centre-view degeneracy), penalty-arc pole/polar hypotheses
- 🟡 Box-only views (penalty arc + one line family) are underdetermined for direct solve — covered by flow propagation; learned-keypoint model is the upgrade path (architecture + training script provided, no bundled weights)
- ✅ Temporal machinery — HSV-histogram scene-cut detector, point-space homography smoothing with mirror-symmetry canonicalisation, optical-flow propagation between/instead of full estimates
- ✅ Manual calibration API (named-keypoint clicks → H)
- ✅ Synthetic ground truth — agent-based match simulator (formations, pressing, man/zonal marking, pass decisions w/ openness scoring, halves + direction flip) + broadcast renderer (true 3D pinhole camera, pan/zoom, GT homographies exact to 0.000 m, GT boxes)
- Validated on synthetic broadcast: centre-circle views calibrate to **0.02–0.27 m** vs ground truth
- ⬜ Tactical radar synced to video (Phase 5 web app)
- 🔄 IN PROGRESS: full-pipeline positional-error measurement (detector→tracker→calibration→table)

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
- ⬜ Off-ball run detection & valuation
- ⬜ RT-DETR vs YOLO benchmark harness

## Environment notes
- Dev env: Windows, Python 3.13 venv at `.venv`; core+app+ml deps installed; torch CPU + ultralytics installed; easyocr NOT installed (heavy) — jersey numbers null in dev runs
- OpenCV 5.0 gotcha discovered: `fitEllipse` rejects non-contiguous arrays (sliced views) — all subsample sites wrap with `ascontiguousarray`
