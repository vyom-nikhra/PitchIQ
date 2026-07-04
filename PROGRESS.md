# PROGRESS

Component truth table. Legend: ✅ fully functional · 🟡 functional with documented fallback/limitation · 🔧 provided, needs user resources · ⬜ open

## Phase 0 — Foundations ✅
- ✅ Config system (pydantic + YAML deep-merge, cache-invalidating config hash)
- ✅ Core models: pitch (33 keypoints), tracking-table schema + parquet IO, MatchMeta, ArtifactStore, .env secret loader (keys never in code/commits)
- 🟡 Data loaders: Metrica / StatsBomb / SoccerNet / Roboflow — wired with legal guardrails; downloads on demand (`scripts/download_data.py`)
- 🟡 Detection: YOLOv11 + RT-DETR behind one interface; COCO fallback (GK/ref via heuristics) until `scripts/train_detector.py` is run; blob fallback for synthetic/no-torch
- ✅ Tracking: self-contained ByteTrack + appearance + camera-motion compensation; native MOTA/IDF1
- ✅ Team assignment (kit clustering + GK/ref heuristics + separability score)
- 🟡 Jersey OCR: easyocr backend + track voting (easyocr optional install)

## Phase 1 — Homography + tracking table ✅ (the crux)
- ✅ Line + conic calibration (component RANSAC, circle-tangency & arc pole/polar constructions), bidirectional scoring, degeneracy gates, chamfer refinement (polish-only), scene cuts, mirror canonicalisation, flow propagation, incumbent-vs-fresh comparison
- ✅ Manual calibration API; 🔧 learned keypoint model (net + local-GPU training script; NDA data → weights not bundled)
- ✅ Synthetic GT harness: agent simulator + 3D-camera broadcast renderer (exact GT homographies/boxes)
- **Measured: 0.25 m direct / 0.72 m median all-frames / 1.28 m full pipeline**
- 🟡 Box-only views ride flow propagation (bounded drift; keypoint model is the fix)

## Phase 2 — Core analytics ✅
Kinematics · possession (0.647 vs 0.650 GT) · heatmaps/territory · formations via Hungarian templates (exact recovery + morphs) · Voronoi + velocity-aware pitch control · line height/compactness · field tilt

## Phase 3 — Intelligence ✅
- ✅ Style features (phase-conditioned) · handcrafted embeddings (robust scale + PCA + group attribution)
- ✅ **Learned contrastive encoder trained** (160 sim pairs, NT-Xent) — demo uses it; FAISS index active
- ✅ Role discovery + naming + nominal-vs-actual flags (scripted archetypes recovered)
- ✅ Similar-player search w/ per-group "why" (cross-team analogues found)
- ✅ Marking: Hungarian timeline, stability, residual-position coupling, GK exclusion — **pairs 10/10 vs GT; man 0.81 vs press-zonal 0.67**

## Phase 4 — Advanced analytics + report ✅
- ✅ Pass detection (R 0.93 vs GT) · pass networks + centralities · line-breaking passes · xT (damped value iteration + attribution) · pressing/PPDA · counters · phase segmentation
- ✅ Grounded report: **Gemini primary (live-verified)**, Anthropic optional, deterministic template fallback; metrics appendix; grounded Q&A w/ retrieval fallback

## Phase 5 — Web app + deployment ✅
- ✅ FastAPI (async jobs, artifact serving, Q&A endpoint) · Streamlit 5-tab dashboard — **all tabs verified rendering in a live browser session**
- ✅ Video-synced tactical radar (canvas + embedded positions; standalone fallback)
- ✅ Bundled GT demo match (learned embeddings, Gemini report, preview media)
- 🔄 CV-variant demo (full perception on rendered video) — building in background
- ✅ Dockerfile (single container, no baked secrets) + compose + HF Spaces config
- ✅ README + docs (architecture / calibration / limitations / data+NDA / deployment / training)
- ⬜ validation.md regeneration via `scripts/validate_synthetic.py` (queued behind CV build)

## Phase 6 — Stretch
- ✅ RT-DETR vs YOLO benchmark harness (`train_detector.py --benchmark`)
- 🔧 Detector fine-tune (Roboflow key ready; local RTX 3050 or Kaggle)
- 🔧 Pitch-keypoint training (SoccerNet download + local GPU)
- ⬜ Team-style fingerprint · off-ball run valuation · jersey OCR end-to-end demo

## Operational notes
- Secrets in `.env` (gitignored; `.env.example` template); NDA data paths hard-blocked in git; keys should be rotated post-project
- Known gotchas: OpenCV 5 `fitEllipse` needs contiguous arrays; PowerShell pipes CRLF-mangle native-command stdin (don't trust `check-ignore --stdin` there)
- Git history rewritten pre-publish to drop accidentally-tracked heavy media (repo was never pushed)
