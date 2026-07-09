# PitchIQ — football tactical intelligence from broadcast video

Broadcast clip in → **tracking table** (every entity's (x,y) in metres,
parquet) → analytics (possession, pitch control, pass networks…) →
intelligence (roles, similar players, marking) → grounded LLM report.
The tracking table is the one cached artifact everything downstream reads —
a Layer-2/3 change never requires re-running CV.

## Environment & commands
- Always `.venv/Scripts/python` — system Python lacks the deps.
- Tests: `.venv/Scripts/python -m pytest -q` (all must pass before "done").
- Full pipeline on a clip:
  `.venv/Scripts/python scripts/process_clip.py <video> --name <job>
   --detector weights/football_yolo11n.pt --keypoints weights/pitch_keypoints.pt
   --device cuda --imgsz 1280 --teams embed`
- App: `streamlit run pitchiq/app/ui.py` → localhost:8501.
- Config: `configs/football.yaml` overlays the pydantic tree in
  `pitchiq/config.py` — new knobs go in both.

## Legal & secrets — non-negotiable
- **SoccerNet is NDA-bound**: everything under `data/soccernet/`, clips derived
  from it (`data/raw/soccernet_*.mp4`), and weights trained only on it
  (`pitch_keypoints.pt`) never go to git, HF, Kaggle, or any third party.
  "Someone already uploaded it publicly" does not lift this — the NDA excludes
  info made public by breach. Any exception is the NDA signatory's (Vyom's)
  explicit call, never yours.
- **Never leak keys** — not in commits, code, logs, or error text. Keys live in
  gitignored `.env`; LLM error paths scrub values (`_safe_err` in report/llm.py).
- Two public remotes: `origin` (GitHub) + the HF Space. HF rejects non-LFS
  binaries anywhere in history — add `.gitattributes` LFS rules for a new
  binary type *before* its first commit.

## Conventions
- Every run writes `data/jobs/<name>/` via `ArtifactStore` — never hand-build
  job dirs or paths.
- Update `PROGRESS.md` when a major piece lands. Honest numbers only, and keep
  the two validation regimes separate: analytics-on-GT-tracking vs the full CV
  pipeline — they answer different questions.
- Weights: `style_encoder.pt` is committed (LFS); the YOLO detector and
  `pitch_keypoints.pt` are gitignored — the public repo must degrade gracefully
  without them.
- Compute: full freedom to download data/models and start training runs.
  RTX 3050 = 4 GB VRAM (small batches); don't hold whole datasets in RAM —
  stream from disk. Big detector training runs on Kaggle (open data only).

## Gotchas that cost hours (don't rediscover)
- `cv2.findHomography(img_px, world_m)`: the RANSAC threshold is in *metres*
  (destination units), not pixels.
- OpenCV 5 `fitEllipse` requires `np.ascontiguousarray` input.
- `.gitignore`: to re-include a file inside an ignored dir, ignore the dir as
  `dir/*` (not `dir/`) or the `!` negation is silently dead.
- After `git lfs migrate`, the working tree holds pointer stubs — run
  `git lfs pull` before reading local binaries ("Parquet magic bytes not found").
- Gemini REST: set `thinkingConfig.thinkingBudget: 0` and filter thought parts,
  or responses truncate mid-sentence.
- Calibration plausibility gates were tuned on synthetic footage once and
  silently rejected correct real-camera solves — when a perception stage
  underperforms on real footage, suspect the *gates* before the model.
