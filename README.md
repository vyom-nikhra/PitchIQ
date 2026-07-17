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

[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20demo-Hugging%20Face%20Space-blue)](https://huggingface.co/spaces/NuclearPanda/PitchIQ)
[![CI](https://github.com/vyom-nikhra/PitchIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/vyom-nikhra/PitchIQ/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**One video clip in → tactical intelligence out.** PitchIQ reconstructs every
player's position on a real 105×68 m pitch from ordinary broadcast footage,
layers analyst-grade tactics on top (pressing, pitch control, marking
schemes), then learns higher-order structure — playing roles discovered from
behaviour, similar-player search, who-marks-whom — and writes the analyst
report, grounded in the computed numbers.

![PitchIQ demo — annotated tracking with persistent IDs, team colours and a live mini-radar](docs/media/demo.gif)

*The bundled demo match: every box, ID, team colour and radar dot computed by
the pipeline. Try it live — **[the demo Space](https://huggingface.co/spaces/NuclearPanda/PitchIQ)**
lands on this match with all five tabs ready.*

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
git clone https://github.com/vyom-nikhra/PitchIQ.git && cd PitchIQ
python -m venv .venv && .venv/Scripts/activate     # (or source .venv/bin/activate)
pip install -e .[app,ml]
pip install torch --index-url https://download.pytorch.org/whl/cpu

python scripts/build_demo.py        # bundled demo match end-to-end (~10 min, once)
streamlit run pitchiq/app/ui.py     # → http://localhost:8501
```

or just: `docker compose up --build`. Add `GEMINI_API_KEY` to `.env`
(template: `.env.example`) for LLM reports & Q&A — everything else works
without any keys. The fine-tuned football detector
([public model repo](https://huggingface.co/NuclearPanda/pitchiq-football-yolo11n))
downloads automatically on first use.

## The dashboard

| Tab | What you get |
|---|---|
| 🎬 Annotated video | boxes, persistent IDs, team colours, jersey numbers, mini-radar |
| 🗺️ Tactical map | **top-down radar frame-synced to video scrubbing** — with motion trails, team-shape hulls and pass arrows, all toggleable |
| 📊 Analytics | possession flow, heatmaps, pitch control, pass networks, formations & shape morphs, xT, phases — tracking/events downloadable as CSV |
| 🧠 Intelligence | discovered role per player (incl. pose-informed style), nominal-vs-actual mismatches, similar-player search with *why*, man/zonal marking with pairs |
| 📝 Report | grounded analyst write-up + "ask the match" Q&A (downloadable) |

Every match header also shows a **tracking-confidence badge** (high / medium /
low) with the perception-quality numbers behind it — see the next section.

## The system knows what it doesn't know

Computer vision on broadcast footage is imperfect, so PitchIQ measures its own
perception quality per clip and **propagates that uncertainty into everything
it says**:

- `data_quality` in the fact base grades calibration coverage, observed-ball
  share, identity stability and kit separability from the cached artifacts;
- the **report hedges when the data is thin** — "possession roughly 60/40 —
  low confidence, ball observed in only 41% of frames" — instead of laundering
  perception noise into confident prose (and stays crisp when quality is high);
- the dashboard badge shows the same assessment, with every number visible;
- a **groundedness audit** (`scripts/audit_report.py`) verifies that every
  numeric claim in a generated report exists in `facts.json` — both bundled
  demo reports audit **100% grounded**, and the check runs in the test suite.

## Validated, not vibes

The repo bundles a **synthetic ground-truth harness**: an agent-based match
simulator (formations, pressing profiles, man/zonal marking with known
assignments) plus a broadcast renderer with a true 3D camera. This separates
two questions most projects blur together — and the split matters, because
they get very different answers:

**(a) Is the analytics *maths* right?** — Layer-2/3 logic scored on
ground-truth tracking:

| What | Result vs ground truth |
|---|---|
| Possession share | 0.647 vs 0.650 true |
| Pass detection | precision 0.90 / recall 0.91 |
| Formations | exact recovery (incl. in/out-of-possession morphs) |
| Who-marks-whom | **10/10 pairs** recovered |
| Man vs zonal separation | 0.81 (man) vs 0.67 (press-heavy zonal) |

**(b) Calibration accuracy** (the crux): 0.25 m mean on direct estimates,
0.72 m median across all frames including flow-bridged ones.

**(c) What does CV noise cost end-to-end?** — the *full CV pipeline*
(detection → tracking → calibration → analytics) run on the rendered video
with no ground-truth shortcuts (`scripts/validate_synthetic.py`, deterministic
seeded clip → [docs/validation.md](docs/validation.md)): calibration holds
(0.32 m median), possession frame-agreement 44%, pass recall 0.14 at 0.80
precision.

**The plain-English version: 44% is the cost of imperfect vision, not broken
math — the same analytics score 0.65-vs-0.65 possession on clean tracking.**
The gap between (a) and (c) *is* the perception error budget, shown rather
than hidden — and the tracking-confidence badge above puts that same honesty
into the product itself, per clip.

The calibration method is the interesting part: line-family hypothesis search
+ **projective conic constructions** (circle tangency points, penalty-arc
pole/polar) that break the centre-view degeneracy, bidirectional mask
scoring, degeneracy gates, chamfer refinement, and flow-bridged temporal
smoothing. Full story: [docs/calibration.md](docs/calibration.md).

## What runs where (and why)

| Stack component | Local (full) | Public Space | Why the difference |
|---|---|---|---|
| Detector | fine-tuned YOLOv11 | **same** — auto-downloaded from the [public model repo](https://huggingface.co/NuclearPanda/pitchiq-football-yolo11n) | trained on open data (CC-BY) → publishable |
| Pitch calibration | learned keypoint model | line/conic only | keypoint model is SoccerNet-trained → **NDA, never redistributed** |
| Ball | TrackNet heatmap tracker | YOLO+Kalman+ROI | TrackNet weights are SoccerNet-trained → NDA |
| Compute | your GPU | free CPU tier (uploads frame-capped) | deployment cost |

The public demo's *pre-baked* showcase match runs the full pipeline; live
uploads on the Space run the reduced stack above — a **deployment and legal
constraint, not the system's ceiling**. The app says exactly which stack
produced what.

## On real broadcast footage

Synthetic ground truth proves the maths; real footage proves the system. The
full trained stack runs end-to-end on real broadcast clips (measured on a
1080p SoccerNet main-camera clip and a 576p Champions League segment):

| Component | Real-footage result |
|---|---|
| Detection | **YOLOv11 fine-tuned** on football: player mAP50 0.99, GK 0.96, referee 0.98, ball 0.63 · ~20 entities/frame with native classes |
| **Calibration** | the **SoccerNet-trained pitch-keypoint model** anchors the panning camera (100 % of frames calibrated on the 1080p clip: 184 keypoint solves + optical-flow bridging); line-only calibration anchored a handful |
| Ball | **TrackNet-style heatmap tracker trained on real footage**: 3 consecutive frames → ball heatmap; on clips it never saw, 52 % detection / **0 % false positives** / 0.9 px median. In-pipeline (with physics smoothing + interpolation) it covers **95 % of frames**, including airborne balls a box detector misses |
| Team assignment | learned **crop-embedding clustering** (CNN → UMAP → K-Means): separability 6.8 on distinct kits, and it resolved the near-identical RMA-white vs City-sky-blue fixture that colour histograms lumped 13:1 |
| Tracking | persistent IDs with a pitch-space max-speed gate (no identity teleports) and **cross-cut re-ID** — players keep their ID through replays and camera changes |
| Style | optional top-down **pose sampling** (lean, stride, crouch…) feeds the role/similarity embeddings |

The **pitch-keypoint model is the enabling piece**: on real box-camera frames
where line/conic calibration correctly refuses (too few markings), it
localises 17–21 semantic keypoints and solves a plausible homography — turning
a raw broadcast pan into real-pitch coordinates for the radar. Trained locally
on a consumer GPU ([docs/training.md](docs/training.md)); it declines on the
synthetic renderer (a different visual domain) and the pipeline falls back to
line/conic there. Honest gaps — pass recall under CV noise, airborne-ball
projection, off-screen players (partially imputed as decaying ghosts), jersey
OCR at low resolution — are catalogued in
[docs/limitations.md](docs/limitations.md).

## Repository tour

- [docs/architecture.md](docs/architecture.md) — layers, artifacts, module map
- [docs/calibration.md](docs/calibration.md) — the crux, measured
- [docs/validation.md](docs/validation.md) — the full-CV error budget, regenerated per run
- [docs/limitations.md](docs/limitations.md) — the honest gap list (read this)
- [docs/roadmap.md](docs/roadmap.md) — landed improvements (embedding team ID,
  real-footage TrackNet ball, cross-cut re-ID, pose features, off-screen
  ghosts, confidence-aware reporting) + the honest backlog
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
   chosen fallback is always recorded in the artifacts. Expected-missing
   degrades; genuine failures raise.
3. **Grounded generation**: the report/Q&A LLM narrates `facts.json` and
   nothing else; the groundedness audit *measures* that promise, and a
   deterministic template stands in when no key is configured.
4. **Honesty as a feature**: known-hard problems (ball, box-only views,
   congestion) are mitigated, measured, and documented — and the running
   system reports its own per-clip confidence rather than pretending.

## Tests

```bash
python -m pytest    # 106 tests: geometry, conics, tracking (incl. cross-cut
                    # re-ID), calibration, ball training/refinement, pose,
                    # analytics, intelligence, report, perception-quality
                    # grading, report-groundedness audit
```

CI runs the full suite + ruff on Python 3.11 and 3.13;
`constraints.txt` pins the exact deployed dependency set.

## Licence

MIT for all code and the synthetic demo data. External datasets keep their
own terms — notably SoccerNet (NDA; never redistributed here). The published
detector weights are AGPL-3.0 (Ultralytics derivative, trained on CC-BY open
data). Built with Ultralytics, OpenCV, SciPy, NetworkX, scikit-learn, FAISS,
PyTorch, Plotly, FastAPI, Streamlit.
