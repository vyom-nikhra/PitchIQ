# Pitch calibration — the crux

Everything downstream consumes real-pitch coordinates, so per-frame
homography estimation received the most engineering care. This documents the
actual method, its measured accuracy, and its honest failure modes.

## Pipeline per frame

1. **White-mask extraction** — grass gate (HSV) → morphological top-hat
   (shadow-robust thin-bright response) → low-saturation gate → Hough
   segments → collinear merge → two orientation families.
2. **Hypothesis generation**
   - *Lines*: order-preserving assignment of image lines to template line
     families (constant-x: goal lines/box fronts/halfway; constant-y:
     touchlines/box sides). All pairwise extended-line intersections are
     valid correspondences (homographies preserve intersections) → DLT +
     RANSAC per assignment.
   - *Conics*: the centre circle and penalty arcs image as ellipses.
     Component-level RANSAC (seeds from arc-fragment components, AMS fit,
     global inlier scoring) recovers the conic even when the circle is
     chopped by the halfway line. Projectively exact constructions then give
     correspondences that lines cannot:
       - ellipse ∩ halfway ↔ circle top/bottom keypoints
       - tangency points from the halfway×touchline corner ↔ the same
         construction on the world circle (tangency is projective-invariant)
       - penalty arc: pole of the box-front line w.r.t. the conic ↔ the
         world pole (pole/polar duality) + arc∩line points.
     This kills the centre-view degeneracy where every line×line
     intersection lies ON the halfway line.
3. **Scoring** — bidirectional: (a) projected-template coverage of the
   dilated mask, discounted by visible fraction; (b) *explanation*: observed
   mask pixels must map close to SOME template marking (KD-tree in metres).
   Wrong assignments cover the few visible lines but leave observed markings
   unexplained — (b) is what makes sparse views discriminative.
4. **Degeneracy gates** — mapped image-corner quad must be simple,
   consistently oriented, with believable area; multi-point local scale must
   be sane and consistent. These reject the collapsed-homography pathology
   that otherwise *maximises* naive scores (found the hard way; see git
   history).
5. **Chamfer refinement** — LM over the 8 parameters against the mask's
   distance transform (soft-L1). Polish only: acceptance is decided on the
   raw score so refinement can never promote a wrong assignment.
6. **Temporal layer** — HSV-histogram scene-cut reset; smoothing in *point
   space* (EMA on four projected control points, exact 4-point refit) with
   mirror-symmetry canonicalisation; optical-flow propagation between/instead
   of full estimates; fresh estimates must beat the flow-propagated incumbent
   on the same frame's evidence before replacing it.

## Measured accuracy (synthetic broadcast, exact ground truth)

| condition | positional error |
|---|---|
| direct estimates (circle/lines visible) | **0.25 m mean** |
| all frames incl. flow-propagated | 0.72 m median / ~2 m mean |
| full pipeline (detector+tracker+calibration) | 1.28 m median |

(Professional multi-camera optical tracking is typically quoted at ~0.5–1 m.)

## Honest failure modes

- **Box-only views** (penalty arc + one line family): all line intersections
  are collinear and the partial-arc conic is too noisy for a stable direct
  solve → those frames ride on flow propagation with bounded drift (worst
  observed ~3–5 m during long structure-poor spells). *Upgrade path:* the
  learned keypoint model (`scripts/train_pitch_keypoints.py`, trains on
  SoccerNet-Calibration) plugs into the same calibrator and takes priority
  when weights are configured.
- **Mirror ambiguity**: the marking template is symmetric, so a single frame
  cannot distinguish a view from its 180° twin. Resolved by temporal
  canonicalisation; per-clip orientation is conventional, not absolute.
- **Hard cuts into structure-poor views** stay uncalibrated (reported as
  `none`, never guessed) until structure returns.
- **Airborne balls** project incorrectly through the ground-plane homography
  — inherent to single-camera systems.
- Real-broadcast extras (worn lines, extreme zoom, rain/glare) are only
  qualitatively covered; the manual-calibration API is the escape hatch.
