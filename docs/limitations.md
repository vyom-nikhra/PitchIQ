# Honest limitations

The system is validated quantitatively against synthetic ground truth and
qualitatively on real footage. What follows is the truthful gap list, ordered
by impact.

1. **Ball tracking is the weakest perception link** (by design acknowledgment,
   not surprise): small, fast, motion-blurred, occluded, airborne. Mitigated
   with Kalman gating, ROI re-inference and gap interpolation (interpolated
   rows carry decayed confidence); airborne balls additionally project wrong
   through the ground-plane homography. Everything ball-derived (possession,
   passes, xT) inherits this noise on real footage.
2. **Box-only camera views** lack direct *line* calibration (collinear
   constraints). **Resolved for real footage** by the trained pitch-keypoint
   model (`scripts/train_pitch_keypoints.py`, SoccerNet-Calibration): on a
   held-out validation split it localises 100% of derivable keypoints at
   2.9 px median, and on real broadcast frames that previously failed
   entirely it now solves cleanly (17–21 keypoints, plausible homographies).
   The model is trained on *real* footage, so on the synthetic renderer it
   (correctly) declines and the pipeline falls back to line/conic
   calibration, which already achieves 0.25 m there. Without the model,
   box-only views ride optical-flow propagation with bounded drift.
3. **Detector**: a YOLOv11n fine-tuned on the Roboflow football set
   (`weights/football_yolo11n.pt`) emits native ball/goalkeeper/player/
   referee classes — validation mAP50 0.89 (player 0.99, GK 0.96, ref 0.98,
   ball 0.63). The COCO-person fallback (GK/referee via colour+position
   heuristics) and the synthetic-only blob detector remain as graceful
   degradation when no weights are supplied.
4. **Team assignment** — **substantially improved** with a learned embedding
   backend (`teams.method: embed`). Colour-histogram clustering collapsed on
   near-identical kits (Real Madrid white vs Man City sky-blue at 576p lumped
   ~13 vs 1 per frame, separability ~1.7). A torchvision-CNN crop embedding
   (→ UMAP → K-Means, the Roboflow-`sports` recipe; SigLIP auto-used if
   `transformers` is installed) separates them: on the same clip, separability
   **1.7 → 3.1**, per-frame split **~13:1 → ~8:5**, and downstream possession
   recovered from a 97%/3% artefact to a realistic **62%/38%**. Residual: a
   mild home-tilt and some players binned to `none`; very similar kits are
   still harder than distinct ones, and the separability score remains
   surfaced in `meta.extras` as a confidence flag. Colour mode
   (`kmeans_lab`) stays the dependency-free default; `embed` is enabled in
   `configs/football.yaml`.
5. **Tracking through congestion** (corners, goalmouths): ByteTrack +
   appearance recovers most occlusions, but long same-kit overlaps still
   cause ID switches; jersey OCR (when enabled) re-anchors identities only
   when digits are legible.
6. **Derived events are approximations**: passes come from possession-spell
   stitching (P≈0.6–0.7 strict / R≈0.9 vs synthetic GT), not a human event
   feed. Deflections and first-time flicks blur attribution.
7. **Phase segmentation is rule-based**; transition shares run high on
   turnover-heavy games. A learned classifier over tracking features is the
   documented upgrade.
8. **Pitch control is simplified Spearman** (arrival-time logistic; no ball
   flight time or control-duration integral).
9. **xT on a single short clip is sparse** — priors dominate; feed longer
   footage or a StatsBomb-fitted grid (`xt.grid_path`) for sharper values.
10. **Style embeddings / roles need minutes**: under ~15–20 min of on-ball
    context per player, role discovery reflects position more than style.
    Synthetic validation shows archetypes separate; real-world validation at
    scale (players recurring across matches) is future work.
11. **Marking scores are a spectrum**: aggressive pressing reads as
    man-oriented (correctly, arguably) — the score separates schemes (0.81
    man vs 0.67 press-zonal on synthetic GT) but the label thresholds are
    conventions.
12. **Throughput**: CPU-only perception runs ~2–4 fps (blob) and slower with
    YOLO; the public demo Space serves precomputed matches and accepts short
    uploads only. GPU deployment removes this.
