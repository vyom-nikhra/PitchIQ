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
2. **Box-only camera views** lack direct calibration (collinear
   constraints); they ride on optical-flow propagation with bounded drift
   (~3–5 m worst observed on synthetic). Fix: train the pitch-keypoint model.
3. **COCO detector fallback** (until you fine-tune on the Roboflow set):
   goalkeepers and referees arrive labelled "person" and are recovered by
   colour-outlier + position heuristics, which can mislabel in crowded
   goalmouths. The blob detector variant is for synthetic renders only.
4. **Team assignment** assumes two dominant kit colours; very similar kits
   lower the separability score (reported in meta) and may flip individual
   tracks under heavy shadow.
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
