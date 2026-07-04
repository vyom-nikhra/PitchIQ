# Architecture

PitchIQ is stratified into three layers; each consumes only the layer below.
The load-bearing design decision: **Layer 1 emits one artifact — the tracking
table — and everything above it is pure dataframe analysis.** Re-running
analytics or intelligence never touches video.

```
video ──► LAYER 1 (perception) ──► tracking.parquet + homography.parquet + meta.json
                                          │
                                          ▼
          LAYER 2 (analytics) ──► kinematics/possession/events/formations/
                                  pitch-control/xT/pressing/phases artifacts
                                          │
                                          ▼
          LAYER 3 (intelligence) ─► embeddings/roles/similarity/marking
                                          │
                                          ▼
          REPORT (grounded LLM) ─► facts.json → report.md (+ Q&A)
```

## Module map

```
pitchiq/
├── config.py                 pydantic tree ← configs/*.yaml (deep-merged)
├── core/                     pitch model (33 keypoints), schema/parquet IO,
│                             projective geometry, video IO, artifact store,
│                             formation templates, .env loader
├── io/                       Metrica / StatsBomb / SoccerNet / Roboflow loaders
├── perception/
│   ├── detection/            YOLOv11 & RT-DETR (one interface) · blob fallback
│   │                         · ball selector (KF gating + ROI + interpolation)
│   ├── tracking/             self-contained ByteTrack · appearance embedders
│   │                         · camera-motion (GMC) · MOTA/IDF1 metrics
│   ├── teams/                kit-colour clustering + GK/referee heuristics
│   ├── jersey/               OCR + confidence-weighted track voting
│   └── calibration/          lines · conics · estimate (hypothesis search)
│                             · refine (chamfer) · temporal · keypoints · manual
├── analytics/                one module per metric family (see docstrings)
├── intelligence/             features · embeddings · encoder · roles ·
│                             similarity · marking
├── report/                   facts · llm providers · generator · qa
├── pipeline/                 perception / analytics / intelligence / report /
│                             full (orchestrates + progress + caching)
├── viz/                      pitch_plot · charts · annotate · radar_html
├── demo/                     match simulator + broadcast renderer (ground truth)
└── app/                      FastAPI backend · Streamlit dashboard
```

## The tracking table

One row per entity per frame (parquet):

```
frame │ timestamp │ entity_id │ class │ team │ jersey_no │
x_pixel │ y_pixel │ x_pitch │ y_pitch │ conf
```

plus `homography.parquet` (flattened 3×3 per frame + reprojection error +
method + scene-cut flag) and `meta.json` (fps, teams, kit colours, attacking
directions, halftime frame). Coordinates: metres on a 105×68 pitch, x∈[0,105],
y∈[0,68]; per-team attacking direction lives in meta and analytics normalise
with `to_attacking_coords`.

## Artifact store

Every processed match is one directory (`data/jobs/<id>` or `data/demo/<name>`)
with a fixed layout (see `core/artifacts.py`). The Streamlit app, the FastAPI
service and the scripts all read/write through `ArtifactStore`, so a match
processed anywhere is served everywhere. `status.json` carries stage/progress
for the async UX.

## Caching & resumability

- Perception is the only expensive stage; its parquet outputs are the cache.
- `Config.config_hash()` fingerprints perception-relevant settings so callers
  can detect stale caches.
- `pipeline/full.py` maps stages onto one progress bar and marks
  `state=error` with the exception on failure — jobs never hang silently.

## Swappability

Detection (`yolo`/`rtdetr`/`blob`), appearance (`osnet`/`colorhist`),
calibration (`keypoints`/`lines`/`manual`), embeddings (`learned`/
`handcrafted`), similarity backend (`faiss`/`sklearn`) and the LLM provider
(`gemini`/`anthropic`/`none`) are all config-selected behind stable
interfaces, each with a documented fallback so the end-to-end path never
breaks on a missing dependency.
