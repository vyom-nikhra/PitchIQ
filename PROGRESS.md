# PROGRESS

Running status of every PitchIQ component.

Legend: ✅ fully functional · 🟡 functional with documented fallback/limitation · 🔧 built, not yet validated · ⬜ not started

## Phase 0 — Foundations ✅
- ✅ Scaffolding, pydantic+YAML config system, config-hash cache invalidation
- ✅ Core domain models (pitch w/ 33 keypoints, tracking-table schema + parquet IO, MatchMeta, ArtifactStore)
- 🟡 Data loaders — Metrica / StatsBomb / SoccerNet / Roboflow parsers wired; downloads need network/keys/NDA (not bundled)
- 🟡 Detection — YOLOv11 & RT-DETR behind one interface (torch CPU + ultralytics installed); no football fine-tune yet → COCO person/sports-ball fallback; colour-blob fallback works on synthetic renders without torch
- ✅ Tracking — self-contained ByteTrack + appearance hooks + camera-motion compensation; MOTA/IDF1/ID-switch metrics
- ✅ Team assignment (LAB+hue clustering, GK/referee heuristics, kit colours)
- 🟡 Jersey OCR — easyocr backend + track-level voting; easyocr not installed in dev env → numbers null

## Phase 1 — Homography + tracking table ✅
- ✅ Line extraction → orientation families; order-preserving template assignment + DLT/RANSAC
- ✅ Conic calibration: ellipse RANSAC (AMS), circle∩halfway + tangent-point constructions (fixes centre-view degeneracy), penalty-arc pole/polar
- ✅ Bidirectional mask scoring + degeneracy gates (corner-quad area/simplicity, multi-point scale) — this fixed catastrophic wrong-side acceptance
- ✅ Chamfer refinement (polish-only, never promotes)
- ✅ Temporal: scene cuts, point-space smoothing w/ mirror canonicalisation, flow propagation, incumbent-vs-fresh evidence comparison
- ✅ Manual calibration API
- ✅ Synthetic ground truth: agent simulator (formations, pressing, man/zonal marking, halves) + broadcast renderer (exact GT homographies & boxes)
- **Measured on rendered broadcast: direct estimates 0.25 m mean; overall median 0.72 m; full pipeline (detect+track+calibrate) median 1.28 m**
- 🟡 Box-only views (penalty arc + one line family) underdetermined for direct solve → bounded flow-drift (worst chunks 3–5 m); learned keypoint model is the upgrade path (net + training script provided, no weights)

## Phase 2 — Core analytics ✅ (53 tests green)
- ✅ Kinematics (Savitzky-Golay, sprints, HI distance) · possession w/ hysteresis (matches GT share 0.647 vs 0.65)
- ✅ Heatmaps + third occupation · formation detection via Hungarian template match (recovers sim formations exactly, in/out-possession morphs)
- ✅ Voronoi (mirror-clipped) + velocity-aware pitch control · line height/compactness · field tilt

## Phase 3 — Intelligence 🔧 (built end-to-end; validation partially done)
- ✅ Player-style features (spatial/movement/involvement/interaction/phase-conditioned, attacking-frame heatmaps)
- ✅ Handcrafted embeddings (robust scaling + PCA, group slices for attribution)
- 🔧 Learned contrastive encoder (SimCLR NT-Xent over 3-channel phase heatmaps) — code + training path done, **no trained weights yet**; auto-falls back to handcrafted
- ✅ Role discovery (silhouette-k KMeans + interpretive naming) — clusters behaviourally coherent on sim; scripted "pressing midfielder" correctly flagged as role/slot mismatch
- ✅ Similar-player search (FAISS w/ sklearn fallback + per-group attribution) — finds cross-team positional analogues
- 🟡 Marking analysis — **who-marks-whom pairs recover 10/10 vs sim ground truth**; man-vs-zonal *scores* did not separate with velocity correlation; rewritten to residual-position coupling (dense, centroid-removed) — **re-validation pending** (next step)

## Phase 4 — Advanced analytics + report (analytics half done in Phase 2 batch)
- ✅ Pass detection (recall 0.93 vs sim GT) · pass networks + centralities · line-breaking passes
- ✅ xT (value iteration w/ distance-decay forward prior + completion damping) · PPDA/pressing · counter-attacks · phase segmentation
- ⬜ LLM analyst report (grounded) + Q&A — template fallback planned; Anthropic API path needs key
- ⬜ Report validation harness

## Phase 5 — Web app + deployment ⬜
- ⬜ FastAPI async jobs · Streamlit 5-tab dashboard · video-synced radar (HTML/canvas component)
- ⬜ Annotated video + radar renderers (viz module)
- ⬜ Bundled pre-computed demo match · Docker/compose · README + architecture docs · validation report

## Phase 6 — Stretch ⬜
- Team-style fingerprint · off-ball run valuation · RT-DETR vs YOLO benchmark harness · detector fine-tune on Roboflow data

## Environment notes
- Windows, Python 3.13 venv `.venv`; core+app+ml+torch(CPU)+ultralytics installed; easyocr NOT installed
- OpenCV 5.0: `fitEllipse` rejects non-contiguous arrays — all subsample sites use `ascontiguousarray`
- Known open items: marking-score re-validation; transition-phase share runs high (rule-based simplification, documented)
