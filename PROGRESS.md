# PROGRESS

Component truth table. Legend: ✅ fully functional · 🟡 functional with documented fallback/limitation · 🔧 provided, needs user resources · ⬜ open

## Phase 0 — Foundations ✅
- ✅ Config system (pydantic + YAML deep-merge, cache-invalidating config hash)
- ✅ Core models: pitch (33 keypoints), tracking-table schema + parquet IO, MatchMeta, ArtifactStore, .env secret loader (keys never in code/commits)
- 🟡 Data loaders: Metrica / StatsBomb / SoccerNet / Roboflow — wired with legal guardrails; downloads on demand (`scripts/download_data.py`)
- ✅ Detection: YOLOv11 fine-tuned on football (player mAP50 0.99 / GK 0.96 / ref 0.98 / ball 0.63), verified on real CL footage (~19 players/frame, native classes); RT-DETR behind same interface; COCO + blob fallbacks retained for graceful degradation
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

## Phase 5 — Web app + deployment ✅ SHIPPED
- ✅ **LIVE: https://huggingface.co/spaces/NuclearPanda/PitchIQ (public, RUNNING)**
- ✅ **Source: https://github.com/vyom-nikhra/PitchIQ (public, LFS for all binaries)**
- ✅ FastAPI (async jobs, artifact serving, Q&A endpoint) · Streamlit 5-tab dashboard — all tabs verified in live browser
- ✅ Video-synced tactical radar (canvas + embedded positions; standalone fallback)
- ✅ Bundled demos: GT variant + CV-pipeline variant with honest comparison banner
- ✅ Docker (publish hurdles solved: HF requires LFS for ALL binaries across history; gitignore dir-negation trap on weights)
- ✅ GEMINI_API_KEY configured as Space secret (user)
- 🔄 validation.md regeneration running (fixed yardsticks: kit-colour team alignment, symmetric MOT proxies)

## Phase 6 — Trained models (stretch → landed)
- ✅ **Detector fine-tuned** (user-run on Kaggle, T4): YOLOv11n on Roboflow football — player mAP50 0.993, GK 0.956, referee 0.977, ball 0.625; integrated at `weights/football_yolo11n.pt`; verified on real footage (20–21 players/frame, native classes)
- ✅ **Keypoint model v2 trained & ACCEPTED** (RTX 3050, SoccerNet-Calibration, NDA-local): expanded supervision (circle/arc/goal-line constructions + orientation disambiguation) → 457 GT keypoints/frame vs v1's 286; **100% detection / 2.9 px median** on held-out valid split. Decisive: real broadcast frames that v1 failed entirely now solve cleanly (17–21 kps, plausible H at ~46m centre). Solve path hardened (≥6-pt consensus + plausibility + mask-score gates — degenerate exact-fits refuse instead of lying). Correctly declines on synthetic renders (trained on real footage) → line/conic fallback there.
- ✅ **Calibration 320× faster on real footage**: keypoint solves (already consensus/plausibility-gated) were re-rejected by a synthetic-tuned mask gate → forced per-frame line search (0.1 fps). Now trust keypoints; line search only when no keypoint model. Keypoint CNN runs on GPU. Measured 0.1 → 31.9 fps, quality unchanged.
- ✅ **Team assignment: whitened K-Means** (standardise signature dims) fixes the raw-scale-dimension collapse. Works when kits differ in hue. HONEST LIMIT: near-identical kits at low res (RMA-white vs City-sky-blue, 576p) still lump together (~13 vs 1/frame) — separability surfaced in meta flags low confidence; real fix needs re-ID embedder or higher res. Documented.
- ✅ **Real broadcast run COMPLETE** (RMA vs MC, 50s, 193s on GPU): detection excellent, keypoint calibration solves the pan (464 keyframes vs 14 for line-only), radar projects real positions. Flagship annotated frame captured. Team colour is the one weak spot (above).
- ✅ Deployment fix: `git lfs migrate` had left demo parquets as pointer stubs in the working tree (local app read them as corrupt) → `git lfs pull` restored; LFS content intact in remote so GitHub/HF unaffected. HF Space verified RUNNING/public.
- 🔄 Remaining (mechanical): final synthetic validation numbers → README table · docker smoke · key rotation reminder · push all fixes to HF Space

## Phase 6 — Stretch
- ✅ RT-DETR vs YOLO benchmark harness (`train_detector.py --benchmark`)
- 🔧 Detector fine-tune (Roboflow key ready; local RTX 3050 or Kaggle)
- 🔧 Pitch-keypoint training (SoccerNet download + local GPU)
- ⬜ Team-style fingerprint · off-ball run valuation · jersey OCR end-to-end demo

## Operational notes
- Secrets in `.env` (gitignored; `.env.example` template); NDA data paths hard-blocked in git; keys should be rotated post-project
- Known gotchas: OpenCV 5 `fitEllipse` needs contiguous arrays; PowerShell pipes CRLF-mangle native-command stdin (don't trust `check-ignore --stdin` there)
- Git history rewritten pre-publish to drop accidentally-tracked heavy media (repo was never pushed)
