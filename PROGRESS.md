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

## Phase 7 — Improvements from peer survey (see docs/roadmap.md)
- ✅ **#1 Embedding team assignment** (`teams.method: embed`): torchvision-CNN crop embeddings → UMAP → K-Means (Roboflow-sports recipe; SigLIP auto if transformers present). Fixes near-identical-kit collapse: separability 1.7→3.1, per-frame ~13:1→~8:5, possession 97/3→62/38 on the real RMA-vs-City clip.
- ✅ **#2 TrackNet ball tracker**: heatmap regression over 3 frames (`perception/detection/tracknet.py`), integrated w/ fallback; synthetic-validated 97% det / 1.6px. Training supports synthetic + SoccerNet-tracking real GT.
- ✅ **Real-footage ball TrackNet TRAINED** (RTX 3050, SoccerNet-tracking test split, NDA-local): 43 train / 6 held-out sequences, bf16, focal loss, early-stopped at epoch 16. Product effect on the SNMOT-187 demo clip: ball coverage 37% → 95% of frames, incl. airborne balls YOLO misses (CAVEAT: 187 is a *training* sequence). Enabled in `configs/football.yaml` (`detection.ball.tracknet_weights`, graceful fallback). NDA: weights stay local/gitignored.
- ✅ **Model selection settled by common-unseen eval** (`scripts/eval_ball_tracker.py`, 4 train-split sequences unseen by both candidates, threshold sweep): local bf16 model **72% det / 27% fp @ 0.35; 51% / 8% @ 0.50; ~0.8 px median** — strictly dominates the user's Kaggle model trained on 2.2× data but in fp16 with overflow clamps (78%/56% @ 0.35; 58%/32% @ 0.50). Honest lesson recorded: numerics (bf16) beat data volume; the fp16 conv clamp that stops T4 BatchNorm poisoning costs accuracy. Product stays on the local model.
- ✅ **Training-infra bugs found the hard way** (all fixed + tested + pushed): fp16 focal-loss saturation → NaN (logsigmoid fp32 form); fp16 activation overflow poisoning BatchNorm buffers via train-mode forwards on skipped batches (bf16 autocast + logit clamp + per-epoch param/buffer health check); dual persistent DataLoader pools OOM'ing 8 GB host RAM (val loads in-process); torch 2.6 `weights_only` rejecting own checkpoint on resume; checkpoint saved before best-score update (stale best on resume).
- ✅ **Calibration fixed on real wide-camera footage** (found via SoccerNet demo clip): keypoint plausibility now checked at inlier keypoints, not extrapolated image corners (12%→80% solve rate); `findHomography` RANSAC threshold was 0.008·width ≈ 15 *metres* (destination units!) → 2.0 m (reproj 311→12 px median). SNMOT-187: 100% frames calibrated, 100% player pitch coverage, possession 67/33.
- ✅ SoccerNet demo clips extracted (`scripts/extract_soccernet_clip.py`, kit-distinctness ranked): SNMOT-187 + SNMOT-149 in data/raw (NDA-local, never commit)

## Phase 8 — Completion sprint (all remaining roadmap items landed)
- ✅ **Ball track refinement** (`detection.ball.postprocess`): per-camera-segment outlier rejection vs rolling median + Savitzky-Golay smoothing, never across scene cuts, before interpolation. SNMOT-187: teleports 31→13, p99 step 197→116 px.
- ✅ **Possession/pass retune for CV error** (swept on the synthetic harness, both regimes): control radius 2→3 m, hysteresis 6→4 → GT pass P/R 0.85/0.88→**0.90/0.91**, full-CV pass recall **0.14→0.40** (P 0.72). Real-clip possession normalised 90/10→63/37.
- ✅ **Cross-cut re-ID** (`tracking.cross_cut_reid`): identities stashed at scene cuts (appearance + last pitch position), claimed back by Hungarian appearance matching with an elapsed-time pitch gate. Unit-tested restore/reject cases.
- ✅ **#3 Pose features** (`pose.enabled`): top-down yolo11n-pose on player crops → 5 scale-free body-shape descriptors → per-track mean/std in pose.parquet → 'pose' group in style embeddings. 12/14 players on a real frame.
- ✅ **#4 Off-screen imputation** (`pitch_control.impute_offscreen`): recently-off-screen players persist as decaying ghosts (≤4 s) for control/Voronoi; residual bias documented (limitations 8b).
- ✅ **UI/UX**: radar motion trails + team-shape hulls + fading pass arrows (toggleable, clock-synced); first-visit tour; CSV/report downloads; kit-colour selector chips.
- ✅ **Config plumbing fix**: `process_clip.py` and local app uploads now default to `configs/football.yaml` (the product config was previously never loaded without explicit flags — pose/imputation/tracknet entries were silently inert).
- ✅ 87 tests green. Final full-stack run on SNMOT-187: 100% frames calibrated, 95% ball coverage, possession 63/37, separability 6.8.

## Phase 9 — Trust & process sprint (post-council review, 2026-07-17)
- ✅ **Confidence-aware reporting** (the council's #1 recommendation): `report/quality.py` grades calibration / observed-ball share / identity stability / kit separability from cached artifacts → `facts["data_quality"]` → the LLM prompt hedges ball-dependent claims when quality is low (was: "no hedging" unconditionally) → dashboard badge with the numbers behind it. Perception now records exact pre-interpolation `ball_observed_frames`.
- ✅ **Report groundedness measured**: `pitchiq/report/audit.py` + `scripts/audit_report.py` verify every numeric claim in a report exists in facts.json (rounding/percent tolerant). Both bundled demo reports: **100% grounded** (43/43 and 49/49 claims). Fabricated-number case unit-tested to fail.
- ✅ **Error discipline**: TrackNet + keypoint-calibrator init now raise typed FileNotFoundError for absent weights; the pipeline catches only expected-missing (FileNotFoundError/ImportError) — a corrupt checkpoint can no longer silently downgrade the crux components.
- ✅ **CI + reproducible deps**: GitHub Actions (ruff + full pytest, py3.11/3.13, CPU torch); next-major caps on every dep; `constraints.txt` freeze wired into the Dockerfile (now python:3.13). Ruff actually passes now (24 pre-existing violations fixed/configured).
- ✅ **Detector published** (owner-approved; open Roboflow CC-BY data, NOT NDA): public HF model repo + `detection.weights_url` auto-download — the public Space and fresh clones get real football detection instead of COCO fallback. Missing weights file now degrades to COCO with a warning (was: crash into blob).
- ✅ **README overhaul**: live-demo link + CI badge + demo GIF at top, real clone URL, what-runs-where table (NDA constraint ≠ capability ceiling), plain-English error-budget sentence.
- ⚠️ **Honest correction**: the Phase 8 claim "full-CV pass recall 0.14→0.40" does NOT reproduce on the standard seeded harness. `validate_synthetic.py` (blob variant, current retuned config, verified twice — deterministic) gives: possession agreement **43.7%** (up from 37.1% pre-retune), pass **P 0.80 / R 0.14**, MOTA 0.528, IDF1 0.249. The 0.40 figure came from the sweep's own evaluation setup; the harness numbers are the ones published in README/docs.
- 106 tests green. Demo reports regenerated (template engine — no GEMINI key in .env at the time; rerun `ReportPipeline` on `data/demo/*` after key rotation to restore Gemini prose).

## Phase 6 — Stretch
- ✅ RT-DETR vs YOLO benchmark harness (`train_detector.py --benchmark`)
- ✅ Detector fine-tuned (Kaggle) · pitch-keypoint model trained (local) · style encoder + ball tracker trained
- ⬜ Team-style fingerprint · off-ball run valuation · jersey OCR end-to-end demo

## Operational notes
- Secrets in `.env` (gitignored; `.env.example` template); NDA data paths hard-blocked in git; keys should be rotated post-project
- Known gotchas: OpenCV 5 `fitEllipse` needs contiguous arrays; PowerShell pipes CRLF-mangle native-command stdin (don't trust `check-ignore --stdin` there)
- Git history rewritten pre-publish to drop accidentally-tracked heavy media (repo was never pushed)
