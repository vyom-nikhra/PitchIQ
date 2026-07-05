---
title: PitchIQ
emoji: ⚽
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# ⚽ PitchIQ — Football Intelligence from Broadcast Video

**One video clip in → research-grade tactical intelligence out.** PitchIQ
reconstructs every player's position on a real 105×68 m pitch from ordinary
broadcast footage, layers analyst-grade tactics on top (pressing, pitch
control, marking schemes), then learns higher-order structure: playing roles
discovered from behaviour, similar-player search, who-marks-whom — and writes
the analyst report, grounded in the computed numbers.

```
┌───────────────────────────────────────────────────────────────────┐
│ LAYER 3 — INTELLIGENCE                                            │
│ style embeddings · role discovery · similar-player search         │
│ marking analysis (man vs zonal, who-marks-whom) · LLM report      │
├───────────────────────────────────────────────────────────────────┤
│ LAYER 2 — ANALYTICS                                               │
│ possession · kinematics · formations & morphs · heatmaps          │
│ Voronoi / pitch control · PPDA & pressing · pass networks         │
│ line-breaking passes · expected threat (xT) · phase segmentation  │
├───────────────────────────────────────────────────────────────────┤
│ LAYER 1 — PERCEPTION                                              │
│ detection (YOLOv11/RT-DETR) · ByteTrack + re-ID + camera motion   │
│ team assignment · jersey OCR · pitch homography (lines + conics)  │
│         → THE TRACKING TABLE: every entity's (x,y) in metres      │
└───────────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
git clone <repo> && cd pitchiq
python -m venv .venv && .venv/Scripts/activate     # (or source .venv/bin/activate)
pip install -e .[app,ml]
pip install torch --index-url https://download.pytorch.org/whl/cpu

python scripts/build_demo.py        # bundled demo match end-to-end (~10 min, once)
streamlit run pitchiq/app/ui.py     # → http://localhost:8501
```

or just: `docker compose up --build`. Add `GEMINI_API_KEY` to `.env`
(template: `.env.example`) for LLM reports & Q&A — everything else works
without any keys.

## The dashboard

| Tab | What you get |
|---|---|
| 🎬 Annotated video | boxes, persistent IDs, team colours, jersey numbers, mini-radar |
| 🗺️ Tactical map | **top-down radar frame-synced to video scrubbing** |
| 📊 Analytics | possession flow, heatmaps, pitch control, pass networks, formations & shape morphs, xT, phases |
| 🧠 Intelligence | discovered role per player, nominal-vs-actual mismatches, similar-player search with *why*, man/zonal marking with pairs |
| 📝 Report | grounded analyst write-up + "ask the match" Q&A |

Upload a clip in the sidebar to run the full pipeline on your own footage
(CPU: expect minutes per video-minute; see `docs/training.md` to unlock the
GPU detector).

## Validated, not vibes

The repo bundles a **synthetic ground-truth harness**: an agent-based match
simulator (formations, pressing profiles, man/zonal marking with known
assignments) plus a broadcast renderer with a true 3D camera — so every layer
is scored against exact truth (`scripts/validate_synthetic.py` →
[docs/validation.md](docs/validation.md)). Highlights:

| What | Result vs ground truth |
|---|---|
| Homography (direct estimates) | **0.25 m** mean positional error |
| Homography (all frames, incl. flow-bridged) | 0.72 m median |
| Full pipeline (detect→track→calibrate) | 1.28 m median |
| Possession share | 0.647 vs 0.650 true |
| Pass detection | recall 0.93 |
| Formations | exact recovery (incl. in/out-of-possession morphs) |
| Who-marks-whom | **10/10 pairs** recovered |
| Man vs zonal separation | 0.81 (man) vs 0.67 (press-heavy zonal) |

The calibration method is the interesting part: line-family hypothesis search
+ **projective conic constructions** (circle tangency points, penalty-arc
pole/polar) that break the centre-view degeneracy, bidirectional mask
scoring, degeneracy gates, chamfer refinement, and flow-bridged temporal
smoothing. Full story: [docs/calibration.md](docs/calibration.md).

## On real broadcast footage

Synthetic ground truth proves the maths; real footage proves the system. The
full trained stack — **YOLOv11 fine-tuned** on football (player mAP50 0.99,
GK 0.96, referee 0.98, ball 0.63) + the **SoccerNet-trained pitch-keypoint
model** + tracking + team clustering — was run end-to-end on a 50-second
Champions League segment (Real Madrid vs Man City, 1024×576):

| Component | Real-footage result |
|---|---|
| Detection | ~19 players + GK + referee + ball per frame, native classes |
| **Calibration** | keypoint model solves the panning camera (464 keyframes + flow); on the same box-camera frames, **line-only calibration anchored just 14** |
| Tracking | persistent IDs across the segment, projected onto a live radar |
| Team colour | separability surfaced in metadata; **fails on this fixture** — RMA-white vs City-sky-blue are near-identical at 576p (documented) |

The **pitch-keypoint model is the enabling piece**: on real box-camera frames
where line/conic calibration correctly refuses (too few markings), it
localises 17–21 semantic keypoints and solves a plausible homography — turning
a raw broadcast pan into real-pitch coordinates for the radar. Trained locally
on a consumer GPU ([docs/training.md](docs/training.md)); it declines on the
synthetic renderer (a different visual domain) and the pipeline falls back to
line/conic there. Honest gaps — team colour on near-identical kits, ball
tracking, jersey OCR at low resolution — are catalogued in
[docs/limitations.md](docs/limitations.md).

## Repository tour

- [docs/architecture.md](docs/architecture.md) — layers, artifacts, module map
- [docs/calibration.md](docs/calibration.md) — the crux, measured
- [docs/limitations.md](docs/limitations.md) — the honest gap list (read this)
- [docs/training.md](docs/training.md) — style encoder (CPU) · YOLO fine-tune
  (local GPU/Kaggle) · pitch keypoints (local GPU, NDA data)
- [docs/data_sources.md](docs/data_sources.md) — licensing, SoccerNet NDA rules,
  secrets policy
- [docs/deployment.md](docs/deployment.md) — Docker · Hugging Face Spaces
- [PROGRESS.md](PROGRESS.md) — component-by-component truth table

## Design principles

1. **The tracking table is the product of Layer 1** — parquet-cached; every
   analytics/intelligence rerun is instant and video-free.
2. **Every component swaps behind config** (detector, calibrator, embedder,
   similarity index, LLM provider) and **degrades gracefully** — the
   end-to-end path survives missing GPUs, keys, or optional deps, and the
   chosen fallback is always recorded in the artifacts.
3. **Grounded generation**: the report/Q&A LLM narrates `facts.json` and
   nothing else; a metrics appendix makes every claim auditable, and a
   deterministic template stands in when no key is configured.
4. **Honesty as a feature**: known-hard problems (ball, box-only views,
   congestion) are mitigated, measured, and documented rather than hidden.

## Tests

```bash
python -m pytest    # 64 tests: geometry, conics, tracking, calibration,
                    # analytics, intelligence, report
```

## Licence

MIT for all code and the synthetic demo data. External datasets keep their
own terms — notably SoccerNet (NDA; never redistributed here). Built with
Ultralytics, OpenCV, SciPy, NetworkX, scikit-learn, FAISS, PyTorch, Plotly,
FastAPI, Streamlit.
