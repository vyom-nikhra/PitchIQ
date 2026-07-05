# HANDOFF — remaining work, decisions made, acceptance gates

Written 2026-07-05 at model-limit boundary. Everything here is executable
without re-deriving context. Read `PROGRESS.md` + memory files first.
Background job still running: **keypoint v2 retrain** (`weights/pitch_keypoints.pt`,
per-epoch checkpoint at `weights/pitch_keypoints.ckpt.pt`).

## State snapshot

- Deployed & public: HF Space (RUNNING) + GitHub. Both remotes must receive
  every push (commands in memory `pitchiq-project-state`).
- Real-clip pre-check (`data/jobs/rma-mc-precheck`, CPU, YOLO@960) findings:
  - Detection: excellent (~19.5 players/frame, GK+ref classes, ball in 1353 frames).
  - **Team clustering FAILED on real kits** (away=20.4k vs home=2.0k rows; kit
    hexes grass-grey). Root cause: 576p → tiny torsos → contaminated signatures.
  - Jersey OCR: 0 reads at 576p — record as measured limitation, stop there.
  - Line calibration anchored only 14 frames (flow carried 1469) — v2
    keypoints are the fix, as designed.
  - 93 tracks in 50 s → ID fragmentation at this resolution (acceptable for
    v1; document).

## Decision 1 — keypoint v2 acceptance gate (task #8, step 1)

When the retrain completes, run `scratchpad/kp_quality.py` (valid-split) and
`scratchpad/kp_eval.py` (real + synthetic frames). Accept v2 iff:
- valid split: detection ≥ 85% AND median localisation ≤ 3 px @480×272;
- real frames (rma_mc_50s): ≥ 2 of the 3 probe frames produce H with
  pitch-plausible centres (inside [-5..110]×[-5..73]).

If it fails: keep v2 anyway ONLY if real-frame solves work (that's the point);
else revert to `git`-untracked v1 (retrain with `--epochs 15` if v1 was
overwritten — checkpoint file may hold a usable model). Do NOT loosen the
consensus gates in `keypoints.py::estimate` to force acceptance — they exist
because degenerate 4-point fits poisoned everything once already.

## Decision 2 — team assignment on real kits (the one real code fix left)

Implement in `pitchiq/perception/teams/assign.py`:
1. In `TeamAssigner.add_sample`: skip boxes with height < 40 px (tiny torsos
   are noise at 576p).
2. In `kit_signature`: for crops where the non-grass pixel count < 60,
   return None (currently 12 — too permissive).
3. Re-weight the signature: chroma-first. Replace
   `[0.5*L, a, b] + 64*hue_hist` with `[0.15*L, 1.5*a, 1.5*b] + 32*hue_hist`
   — white-vs-skyblue separates in b* chroma; L* carries shadows.
4. Add a unit test with synthetic white-kit vs skyblue-kit crops (paint
   40×40 patches, assert 2 clusters + separability > 2).

Iteration loop (fast, after v2 lands — GPU makes perception ~8 min):
```
python scratchpad/real_precheck.py   # edit device: "cuda", add calibration.keypoint_weights
```
Success = home/away row counts within 35–65% of each other, kit hexes
recognisably white-ish and blue-ish, separability > 2. Expected teams:
RMA ≈ white/#eeeeee, City ≈ sky blue/#6cabdd (2026 kits may differ — check a
frame visually, `scratchpad/real_frame.png` exists).

## Decision 3 — final numbers protocol (task #9)

Order matters (each step feeds the next):
1. `python scripts/recalibrate.py data/demo/synthetic-derby-cv --keypoint-weights weights/pitch_keypoints.pt`
   Baseline to beat: 2.93 m median positional error vs the GT twin; 8 passes
   (GT 60); possession 81/19 (GT 65/35). Measure with a decompose-style
   nearest-GT script (see `scratchpad/decompose.py` pattern, GT = synthetic-derby).
   Success: median ≤ 1.5 m and passes ≥ 25. Partial success is fine —
   report honestly in README either way; the GT-vs-CV comparison IS the story.
2. Official validation with the product stack:
   `python scripts/validate_synthetic.py --detector-weights weights/football_yolo11n.pt --device cuda --keypoint-weights weights/pitch_keypoints.pt`
   (script now aligns team labels by position overlap; blob-mode kit colours
   were the reason possession agreement read 42.5% — expect ≥ 75% aligned).
3. Real-clip full run into `data/jobs/rma-mc-full`:
   YOLO cuda + keypoints + jersey off + teams fix from Decision 2.
   Inspect all 5 tabs via `.claude/launch.json` preview. The radar over real
   footage is the flagship screenshot for the README.
4. Rebuild demo GT variant once (`python scripts/build_demo.py --skip-encoder`)
   so bundled phases reflect the tightened transition rule; then re-push BOTH
   remotes (HF rebuild is automatic; verify stage RUNNING after).

## Decision 4 — what NOT to do

- No more line-calibrator tuning for real footage (keypoints own that regime).
- No jersey-classifier training (576p ceiling; documented).
- No SoccerNet on Kaggle, ever (NDA — memory `soccernet-nda-rules`).
- Don't commit `weights/pitch_keypoints.pt` or `weights/football_yolo11n.pt`
  (NDA-derived / user's artifact); only `style_encoder.pt` ships.
- Don't chase MOTA/IDF1 improvements via eval-side changes — the proxy is
  documented as a proxy; real MOT benchmarking = SoccerNet tracking split
  (future work).

## Wrap-up checklist (task #10, all mechanical)

README validation table + real-footage section (numbers from steps above,
screenshot from the app) → PROGRESS final sweep → docker build smoke →
key rotation reminder to user (Gemini/Roboflow/HF fine-grained + revoke old
broad HF token) → final push to both remotes → close tasks #7–#10.
