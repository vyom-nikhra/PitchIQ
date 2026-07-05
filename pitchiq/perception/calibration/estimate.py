"""Line+conic homography estimation: template matching + DLT + RANSAC.

This is PitchIQ's primary calibration path — the crux of the whole system.

**Lines.** The pitch template has two families of marking lines — constant
world-x (goal lines, box fronts, halfway) and constant world-y (touchlines,
box sides). Detected image segments are split into families, ordered
spatially, and we enumerate *order-preserving* assignments of image lines to
template lines. Every hypothesis yields point correspondences (all pairwise
intersections of the extended lines — homographies preserve intersections, so
points off the physical markings are still valid constraints), solved with
``cv2.findHomography(..., RANSAC)``.

**Conics.** Centre-of-pitch views are degenerate for lines alone: every
nameable intersection lies ON the halfway line, so DLT has no off-axis
constraint. The centre circle (imaged as an ellipse) fixes this with
projectively exact constructions (see :mod:`.conics`):

* ellipse ∩ halfway → the circle's top/bottom keypoints,
* tangency points of the tangents from the halfway×touchline corner → the
  matching world tangency points (tangency is projective-invariant),
* for penalty arcs: arc ∩ box-front line → the two arc keypoints, and the
  *pole* of the box-front line w.r.t. the conic → the corresponding world
  pole (pole/polar duality is projective-invariant) — three non-collinear
  correspondences from a single arc.

**Scoring.** Each candidate H projects the full template back into the frame;
the fraction of projected points landing on the (dilated) white-line mask —
weighted by how much template is visible and how many lines were matched —
ranks hypotheses. Plausibility gates (scale, on-pitch centre) reject
nonsense before scoring.

Honest limitations (also in README): views with almost no structure (tight
zooms, midfield sideline shots without the circle) are underdetermined and
rely on optical-flow propagation from the last calibrated frame; those frames
are labelled ``flow`` in the homography table rather than silently wrong.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from pitchiq.core.geometry import apply_homography, line_intersection, point_line_distance
from pitchiq.core.pitch import CIRCLE_RADIUS, PENALTY_AREA_DEPTH, PENALTY_SPOT_DIST, Pitch
from pitchiq.perception.calibration.conics import (
    circle_conic,
    conic_line_intersections,
    ellipse_to_conic,
    line_through,
    pole_point,
    sample_ellipse,
    tangent_points_from,
)
from pitchiq.perception.calibration.lines import Segment, extract_pitch_lines, split_line_families
from pitchiq.perception.calibration.refine import refine_homography

log = logging.getLogger(__name__)


def _template_lines(pitch: Pitch) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    L, W = pitch.length, pitch.width
    pa_y0 = (W - 40.32) / 2
    ga_y0 = (W - 18.32) / 2
    x_lines = [  # constant world-x, ordered left -> right
        ("goal_left", 0.0),
        ("ga_left_front", 5.5),
        ("pa_left_front", 16.5),
        ("halfway", L / 2),
        ("pa_right_front", L - 16.5),
        ("ga_right_front", L - 5.5),
        ("goal_right", L),
    ]
    y_lines = [  # constant world-y, ordered bottom -> top
        ("touch_bottom", 0.0),
        ("pa_bottom", pa_y0),
        ("ga_bottom", ga_y0),
        ("ga_top", W - ga_y0),
        ("pa_top", W - pa_y0),
        ("touch_top", float(W)),
    ]
    return x_lines, y_lines


def plausible_homography(H: np.ndarray, w_img: int, h_img: int, pitch: Pitch) -> bool:
    """Reject (near-)degenerate homographies — shared by every calibration path.

    A collapsed H (whole image → a point neighbourhood on the template)
    maximises naive coverage/explanation scores, so these gates run first:
    the image-corner quad must map to a simple, consistently oriented polygon
    of believable area, and the local metre-per-pixel scale must be believable
    and consistent across the frame.
    """
    corners = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype=float)
    quad = apply_homography(H, corners)
    if not np.all(np.isfinite(quad)):
        return False
    x, y = quad[:, 0], quad[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if not (150.0 <= area <= 4.0 * pitch.length * pitch.width):
        return False
    crosses = []
    for i in range(4):
        a = quad[(i + 1) % 4] - quad[i]
        b = quad[(i + 2) % 4] - quad[(i + 1) % 4]
        crosses.append(a[0] * b[1] - a[1] * b[0])
    if not (all(cr > 0 for cr in crosses) or all(cr < 0 for cr in crosses)):
        return False
    probes = np.array([[w_img * fx, h_img * fy]
                       for fx in (0.25, 0.5, 0.75) for fy in (0.35, 0.65)])
    p0 = apply_homography(H, probes)
    p1 = apply_homography(H, probes + [100.0, 0.0])
    spans = np.linalg.norm(p1 - p0, axis=1)
    if not np.all(np.isfinite(spans)):
        return False
    if spans.min() < 0.3 or spans.max() > 40.0 or spans.max() / max(spans.min(), 1e-9) > 8.0:
        return False
    return True


@dataclass
class Hypothesis:
    H: np.ndarray
    score: float
    reproj_error_px: float
    n_lines: int
    assignment: dict[str, int] = field(default_factory=dict)
    kind: str = "line"  # line | circle | arc | hint — how the correspondences arose
    raw_score: float = -1.0  # pre-refinement score; acceptance gates use THIS


@dataclass
class DetectedEllipse:
    cx: float
    cy: float
    major: float   # full axis length (px)
    minor: float
    angle: float   # degrees, cv2 convention
    n_pts: int

    @property
    def Q(self) -> np.ndarray:
        return ellipse_to_conic(self.cx, self.cy, self.major, self.minor, self.angle)

    def polyline(self, n: int = 120) -> np.ndarray:
        return sample_ellipse(self.cx, self.cy, self.major, self.minor, self.angle, n)


class LineCalibrator:
    """Single-frame homography estimation from pitch lines and circle conics."""

    def __init__(
        self,
        pitch: Pitch,
        max_lines_per_family: int = 4,
        min_line_score: float = 0.25,
        min_seg_len_frac: float = 0.05,
    ) -> None:
        self.pitch = pitch
        self.max_lines = max_lines_per_family
        self.min_line_score = min_line_score
        self.min_seg_len_frac = min_seg_len_frac
        self.x_lines, self.y_lines = _template_lines(pitch)
        self._template_samples = pitch.sample_line_points(step=1.0)
        from scipy.spatial import cKDTree

        self._template_tree = cKDTree(pitch.sample_line_points(step=0.5))
        self._last_mask_dil: np.ndarray | None = None
        self._last_mask_pts: np.ndarray | None = None
        self._last_shape: tuple[int, int] | None = None

    def score_homography(self, H: np.ndarray) -> float:
        """Mask-coverage score of an arbitrary H on the most recently
        estimated frame (call right after :meth:`estimate`). -1 if unscorable."""
        if self._last_mask_dil is None or H is None:
            return -1.0
        hyp = Hypothesis(H=H, score=0.0, reproj_error_px=0.0, n_lines=0)
        scored = self._score(hyp, self._last_mask_dil, self._last_shape)
        return -1.0 if scored is None else float(scored.score)

    # ------------------------------------------------------------------ API
    def estimate(self, frame_bgr: np.ndarray, hint_H: np.ndarray | None = None) -> Hypothesis | None:
        """Estimate the pixel→pitch homography for one frame.

        ``hint_H``: a nearby homography (e.g. flow-propagated). When given,
        line→template assignment is resolved directly through the hint (one
        hypothesis instead of a combinatorial search); the full search still
        runs if the hinted hypothesis scores poorly.
        """
        h_img, w_img = frame_bgr.shape[:2]
        segs, mask, _field = extract_pitch_lines(frame_bgr, self.min_seg_len_frac)
        # cache for score_homography() (lets the caller compare an incumbent
        # solution against a fresh estimate on the same evidence)
        self._last_mask_dil = cv2.dilate(mask, np.ones((9, 9), np.uint8))
        self._last_shape = (h_img, w_img)
        ys_m, xs_m = np.nonzero(mask)
        if len(xs_m):
            mask_pts = np.column_stack([xs_m, ys_m]).astype(np.float64)
            if len(mask_pts) > 400:
                mask_pts = np.ascontiguousarray(mask_pts[:: len(mask_pts) // 400])
            self._last_mask_pts = mask_pts
        else:
            self._last_mask_pts = None
        if not segs:
            return None
        ellipses = self._detect_ellipses(mask, segs, (h_img, w_img))
        segs = self._drop_arc_fragments(segs, ellipses)

        h_fam, v_fam = split_line_families(segs)
        h_fam = sorted(h_fam, key=lambda s: -s.length)[: self.max_lines]
        v_fam = sorted(v_fam, key=lambda s: -s.length)[: self.max_lines]
        # spatial ordering: v-family left->right at mid height, h-family top->bottom
        v_fam = sorted(v_fam, key=lambda s: s.x_at_y(h_img / 2))
        h_fam = sorted(h_fam, key=lambda s: s.y_at_x(w_img / 2))

        dil = self._last_mask_dil

        best: Hypothesis | None = None
        if hint_H is not None:
            hinted = self._hinted_hypothesis(v_fam, h_fam, hint_H, (h_img, w_img))
            if hinted is not None:
                best = self._score(hinted, dil, (h_img, w_img))
                if best is not None and best.score >= max(1.5 * self.min_line_score, 0.4):
                    # confident fast path — acceptance decided BEFORE refinement
                    return self._refine(best, mask, dil, (h_img, w_img))

        candidates = itertools.chain(
            self._line_hypotheses(v_fam, h_fam, (h_img, w_img)),
            self._ellipse_hypotheses(ellipses, v_fam, h_fam, (h_img, w_img)),
        )
        for hyp in candidates:
            scored = self._score(hyp, dil, (h_img, w_img))
            if scored is None:
                continue
            if best is None or scored.score > best.score:
                best = scored
        # acceptance is judged on the RAW hypothesis score; refinement only
        # polishes an already-accepted solution (it must never promote a wrong
        # assignment over the threshold by dragging it onto the mask).
        if best is None or best.score < self.min_line_score:
            return None
        return self._refine(best, mask, dil, (h_img, w_img))

    def _refine(self, best: Hypothesis, mask, dil, img_shape) -> Hypothesis:
        """Chamfer-polish the winning hypothesis; keep only if it scores better.

        ``raw_score`` (pre-refinement) is preserved: acceptance gates must not
        be cleared by refinement pulling a wrong assignment onto the mask.
        """
        if best.raw_score < 0:
            best.raw_score = best.score
        ref = refine_homography(best.H, mask, self._template_samples)
        if ref is None:
            return best
        H1, _cb, _ca = ref
        cand = Hypothesis(H=H1, score=0.0, reproj_error_px=best.reproj_error_px,
                          n_lines=best.n_lines, assignment=dict(best.assignment),
                          kind=best.kind, raw_score=best.raw_score)
        scored = self._score(cand, dil, img_shape)
        if scored is not None and scored.score >= best.score:
            return scored
        return best

    # -------------------------------------------------------------- ellipses
    def _detect_ellipses(
        self, mask: np.ndarray, segments: list[Segment], img_shape
    ) -> list[DetectedEllipse]:
        """Fit ellipses to the curved white structure left after erasing lines.

        Only *long* segments (confident straight lines) are erased: Hough also
        chops circle arcs into short pseudo-segments, and erasing those would
        delete the very evidence we need. The erasure splits a circle into
        several arc components, so components are clustered by proximity and
        each cluster gets one trimmed (outlier-robust) ellipse fit, validated
        by residual and angular span.
        """
        h_img, w_img = img_shape
        clean = mask.copy()
        for s in segments:
            if s.length > 0.15 * w_img:
                cv2.line(clean, (int(s.x1), int(s.y1)), (int(s.x2), int(s.y2)), 0, 15)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
        comps = [
            i for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= 40
            and max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= 0.02 * w_img
        ]
        if not comps:
            return []
        comp_pts = {}
        for i in comps:
            ys, xs = np.nonzero(labels == i)
            p = np.column_stack([xs, ys]).astype(np.float32)
            if len(p) > 400:
                p = np.ascontiguousarray(p[:: len(p) // 400])
            comp_pts[i] = p
        all_pts = np.concatenate(list(comp_pts.values()))
        if len(all_pts) > 1200:
            all_pts = np.ascontiguousarray(all_pts[:: len(all_pts) // 1200])

        # component-level RANSAC: seed fits from single components and pairs,
        # score by inliers over ALL curved points, refit best on its inliers.
        seeds: list[np.ndarray] = [p for p in comp_pts.values() if len(p) >= 60]
        for a, b in itertools.combinations(comps, 2):
            seeds.append(np.concatenate([comp_pts[a], comp_pts[b]]))
        fits = []
        for seed in seeds[:60]:
            if len(seed) < 5:
                continue
            fit = self._fit_and_inliers(seed, all_pts, (h_img, w_img))
            if fit is not None:
                fits.append(fit)
        fits.sort(key=lambda f: -len(f[5]))

        out: list[DetectedEllipse] = []
        for fit in fits:
            refit = self._fit_and_inliers(fit[5], all_pts, (h_img, w_img))
            if refit is not None:
                fit = refit
            cx, cy, MA, ma, ang, inl = fit
            # deduplicate: skip fits whose centre is near an accepted one
            if any(np.hypot(cx - e.cx, cy - e.cy) < 0.2 * w_img for e in out):
                continue
            th = np.arctan2(inl[:, 1] - cy, inl[:, 0] - cx)
            bins = np.unique((np.degrees(th) // 15).astype(int))
            # >= ~60 degrees of subtended arc; degenerate straight-line fits are
            # already rejected by the axis-ratio gate, and hypothesis scoring
            # against the white mask is the final arbiter.
            if len(bins) < 4:
                continue
            out.append(DetectedEllipse(float(cx), float(cy), float(MA), float(ma), float(ang),
                                       len(inl)))
            if len(out) == 2:
                break
        return out

    def _fit_and_inliers(self, seed: np.ndarray, all_pts: np.ndarray, img_shape):
        """Fit an ellipse to ``seed``; return fit + global inliers, or None."""
        h_img, w_img = img_shape
        seed_c = np.ascontiguousarray(seed, dtype=np.float32)
        try:  # AMS is markedly more stable on partial arcs than the default fit
            (cx, cy), (a1, a2), ang = cv2.fitEllipseAMS(seed_c)
            if not all(np.isfinite([cx, cy, a1, a2, ang])):
                raise cv2.error("non-finite AMS fit")
        except cv2.error:
            try:
                (cx, cy), (a1, a2), ang = cv2.fitEllipse(seed_c)
            except cv2.error:
                return None
        MA, ma = (a1, a2) if a1 >= a2 else (a2, a1)
        if a2 > a1:
            ang += 90.0
        if not (0.06 * w_img < MA < 2.5 * w_img) or ma < 0.02 * h_img or MA / max(ma, 1) > 12:
            return None
        poly = sample_ellipse(cx, cy, MA, ma, ang, 200).astype(np.float32)
        d = np.min(np.linalg.norm(all_pts[:, None, :] - poly[None, :, :], axis=2), axis=1)
        inl = all_pts[d < 3.5]
        if len(inl) < 80 or np.median(d[d < 3.5]) > 2.5:
            return None
        return cx, cy, MA, ma, ang, inl

    @staticmethod
    def _drop_arc_fragments(segments: list[Segment], ellipses: list[DetectedEllipse]) -> list[Segment]:
        """Remove Hough fragments that trace an accepted ellipse (arc pieces
        masquerading as short lines pollute the orientation families)."""
        if not ellipses:
            return segments
        polys = [e.polyline(160) for e in ellipses]
        kept = []
        for s in segments:
            probe = np.array([s.midpoint, *s.endpoints()])
            near = False
            for poly in polys:
                d = np.min(np.linalg.norm(probe[:, None] - poly[None], axis=2), axis=1)
                if np.all(d < 14.0):
                    near = True
                    break
            if not near:
                kept.append(s)
        return kept

    # ------------------------------------------------------- hypothesis gen
    def _line_hypotheses(self, v_fam, h_fam, img_shape):
        """Order-preserving line-assignment hypotheses (needs ≥2 per family)."""
        n_v, n_h = len(v_fam), len(h_fam)
        if n_v < 1 or n_h < 1:
            return
        h_img, w_img = img_shape
        v_candidates = list(itertools.combinations(self.x_lines, n_v))
        for flip in (False, True):
            y_templ = self.y_lines if not flip else [(n, self.pitch.width - v) for n, v in self.y_lines]
            # image top -> bottom must map to descending template order
            h_candidates = [
                tuple(reversed(c)) for c in itertools.combinations(y_templ, n_h)
            ]
            for v_assign in v_candidates:
                for h_assign in h_candidates:
                    corr_img, corr_pitch = [], []
                    names = {}
                    for vi, (vname, vx) in enumerate(v_assign):
                        for hi, (hname, hy) in enumerate(h_assign):
                            a1, a2 = v_fam[vi].endpoints()
                            b1, b2 = h_fam[hi].endpoints()
                            pt = line_intersection(a1, a2, b1, b2)
                            if pt is None:
                                continue
                            if not (-1.5 * w_img < pt[0] < 2.5 * w_img and -1.5 * h_img < pt[1] < 2.5 * h_img):
                                continue
                            corr_img.append(pt)
                            corr_pitch.append((vx, hy))
                            names[f"{vname}*{hname}"] = len(corr_img) - 1
                    hyp = self._solve(corr_img, corr_pitch, n_v + n_h, names)
                    if hyp is not None:
                        yield hyp

    def _ellipse_hypotheses(self, ellipses, v_fam, h_fam, img_shape):
        """Conic-based hypotheses: centre-circle and penalty-arc views."""
        L, W = self.pitch.length, self.pitch.width
        cy_w = W / 2
        for e in ellipses:
            Q = e.Q
            center = np.array([e.cx, e.cy])
            for v_seg in v_fam:
                a, b = v_seg.endpoints()
                d_center = float(point_line_distance(center[None], a, b)[0])
                l_img = line_through(a, b)
                cuts = conic_line_intersections(Q, l_img)
                if len(cuts) != 2:
                    continue

                if d_center < 0.30 * (e.minor / 2):
                    # ---- centre circle + halfway line -----------------------
                    world_circle = circle_conic(L / 2, cy_w, CIRCLE_RADIUS)
                    p_top, p_bot = (L / 2, cy_w + CIRCLE_RADIUS), (L / 2, cy_w - CIRCLE_RADIUS)
                    for h_seg in h_fam:
                        c1, c2 = h_seg.endpoints()
                        A_img = line_intersection(a, b, c1, c2)
                        if A_img is None:
                            continue
                        T_img = tangent_points_from(Q, A_img)
                        if len(T_img) != 2:
                            continue
                        for touch_y in (W, 0.0):
                            A_w = np.array([L / 2, touch_y])
                            T_w = tangent_points_from(world_circle, A_w)
                            if len(T_w) != 2:
                                continue
                            for cut_order in (0, 1):
                                for t_order in (0, 1):
                                    corr_img = [A_img, cuts[cut_order], cuts[1 - cut_order],
                                                T_img[t_order], T_img[1 - t_order]]
                                    corr_pitch = [tuple(A_w), p_top, p_bot,
                                                  tuple(T_w[0]), tuple(T_w[1])]
                                    hyp = self._solve(corr_img, corr_pitch, 2,
                                                      {"circle+halfway": -1}, kind="circle")
                                    if hyp is not None:
                                        yield hyp
                elif 0.2 * (e.major / 2) < d_center < 1.4 * (e.major / 2):
                    # ---- penalty arc + box-front line ------------------------
                    arc_dy = np.sqrt(CIRCLE_RADIUS**2 - (PENALTY_AREA_DEPTH - PENALTY_SPOT_DIST) ** 2)
                    pole_img = pole_point(Q, l_img)
                    if pole_img is None:
                        continue
                    for side in ("left", "right"):
                        spot_x = PENALTY_SPOT_DIST if side == "left" else L - PENALTY_SPOT_DIST
                        front_x = PENALTY_AREA_DEPTH if side == "left" else L - PENALTY_AREA_DEPTH
                        world_arc = circle_conic(spot_x, cy_w, CIRCLE_RADIUS)
                        pole_w = pole_point(world_arc, np.array([1.0, 0.0, -front_x]))
                        if pole_w is None:
                            continue
                        i_top, i_bot = (front_x, cy_w + arc_dy), (front_x, cy_w - arc_dy)
                        # intersections of the box-front line with h-family lines
                        # (in image top->bottom order, matching h_fam's sort)
                        extra_img = []
                        for h_seg in h_fam[:3]:
                            c1, c2 = h_seg.endpoints()
                            pt = line_intersection(a, b, c1, c2)
                            if pt is not None:
                                extra_img.append(pt)
                        k = len(extra_img)
                        names = {f"arc_{side}": -1}
                        for cut_order in (0, 1):
                            corr_img = [pole_img, cuts[cut_order], cuts[1 - cut_order]]
                            corr_pitch = [tuple(pole_w), i_top, i_bot]
                            yielded_with_lines = False
                            if k:
                                # order-preserving template-y assignment for the
                                # h-lines, both vertical orientations
                                for flip in (False, True):
                                    y_templ = (self.y_lines if not flip
                                               else [(nm, W - v) for nm, v in self.y_lines])
                                    for combo in itertools.combinations(y_templ, k):
                                        assign = tuple(reversed(combo))  # image top->bottom = descending y
                                        ci = corr_img + extra_img
                                        cp = corr_pitch + [(front_x, hy) for _nm, hy in assign]
                                        hyp = self._solve(ci, cp, 1 + k, names, kind="arc")
                                        if hyp is not None:
                                            yielded_with_lines = True
                                            yield hyp
                            if not yielded_with_lines:
                                hyp = self._solve(corr_img, corr_pitch, 1, names, kind="arc")
                                if hyp is not None:
                                    yield hyp

    def _hinted_hypothesis(self, v_fam, h_fam, hint_H, img_shape) -> Hypothesis | None:
        """Assign each image line to its nearest template line via the hint."""
        h_img, w_img = img_shape
        v_assign: list[tuple[Segment, float]] = []
        for s in v_fam:
            mid = apply_homography(hint_H, s.midpoint[None])[0]
            if not np.all(np.isfinite(mid)):
                continue
            _name, coord = min(self.x_lines, key=lambda t: abs(t[1] - mid[0]))
            if abs(coord - mid[0]) <= 6.0:
                v_assign.append((s, coord))
        h_assign: list[tuple[Segment, float]] = []
        for s in h_fam:
            mid = apply_homography(hint_H, s.midpoint[None])[0]
            if not np.all(np.isfinite(mid)):
                continue
            _name, coord = min(self.y_lines, key=lambda t: abs(t[1] - mid[1]))
            if abs(coord - mid[1]) <= 6.0:
                h_assign.append((s, coord))
        corr_img, corr_pitch = [], []
        for sv, vx in v_assign:
            for sh, hy in h_assign:
                a1, a2 = sv.endpoints()
                b1, b2 = sh.endpoints()
                pt = line_intersection(a1, a2, b1, b2)
                if pt is None:
                    continue
                if not (-1.5 * w_img < pt[0] < 2.5 * w_img and -1.5 * h_img < pt[1] < 2.5 * h_img):
                    continue
                corr_img.append(pt)
                corr_pitch.append((vx, hy))
        return self._solve(corr_img, corr_pitch, len(v_assign) + len(h_assign),
                           {"hint": -1}, kind="hint")

    # ----------------------------------------------------------------- solve
    def _solve(self, corr_img, corr_pitch, n_lines: int, names: dict,
               kind: str = "line") -> Hypothesis | None:
        if len(corr_img) < 4:
            return None
        src = np.asarray(corr_img, dtype=np.float64)
        dst = np.asarray(corr_pitch, dtype=np.float64)
        if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
            return None
        if np.linalg.matrix_rank(np.cov(src.T)) < 2:
            return None
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or inliers is None or inliers.sum() < 4:
            return None
        inl = inliers.ravel() == 1
        err = self._keypoint_reproj_error(H, src[inl], dst[inl])
        return Hypothesis(H=H, score=0.0, reproj_error_px=err, n_lines=n_lines,
                          assignment=names, kind=kind)

    # ---------------------------------------------------------------- score
    def _score(self, hyp: Hypothesis, mask_dilated: np.ndarray, img_shape) -> Hypothesis | None:
        h_img, w_img = img_shape
        H = hyp.H
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        c = apply_homography(H, [[w_img / 2, h_img / 2]])[0]
        if not np.all(np.isfinite(c)):
            return None
        if not (-10 <= c[0] <= self.pitch.length + 10 and -10 <= c[1] <= self.pitch.width + 10):
            return None
        if not self._plausible_geometry(H, w_img, h_img):
            return None

        proj = apply_homography(Hinv, self._template_samples)
        ok = np.isfinite(proj).all(axis=1)
        proj = proj[ok]
        inside = (
            (proj[:, 0] >= 0) & (proj[:, 0] < w_img - 1) & (proj[:, 1] >= 0) & (proj[:, 1] < h_img - 1)
        )
        n_visible = int(inside.sum())
        if n_visible < 60:
            return None
        pts = proj[inside].astype(int)
        on_mask = mask_dilated[pts[:, 1], pts[:, 0]] > 0
        coverage = float(on_mask.mean())
        visible_frac = n_visible / len(self._template_samples)

        # bidirectional consistency: the observed white pixels must in turn be
        # *explained* by the template under H. A wrong line assignment can lay
        # some template onto the few visible lines (high forward coverage) but
        # always leaves detected markings mapping to empty pitch — this factor
        # is what makes sparse-view scoring discriminative.
        explained = 1.0
        if self._last_mask_pts is not None and len(self._last_mask_pts) >= 40:
            world = apply_homography(H, self._last_mask_pts)
            okm = np.isfinite(world).all(axis=1)
            if okm.sum() >= 40:
                d, _ = self._template_tree.query(world[okm], k=1)
                explained = float(np.mean(d < 1.0))

        hyp.score = float(
            coverage * np.sqrt(visible_frac) * (1 + 0.05 * hyp.n_lines)
            * (0.2 + 0.8 * explained)
        )
        return hyp

    def _plausible_geometry(self, H: np.ndarray, w_img: int, h_img: int) -> bool:
        return plausible_homography(H, w_img, h_img, self.pitch)

    def prepare_frame(self, frame_bgr: np.ndarray) -> None:
        """Extract + cache the white-line mask WITHOUT running estimation, so
        :meth:`score_homography` can gate homographies from other sources
        (keypoint model, manual clicks) against the same pixel evidence."""
        h_img, w_img = frame_bgr.shape[:2]
        _segs, mask, _field = extract_pitch_lines(frame_bgr, self.min_seg_len_frac)
        self._last_mask_dil = cv2.dilate(mask, np.ones((9, 9), np.uint8))
        self._last_shape = (h_img, w_img)
        ys_m, xs_m = np.nonzero(mask)
        if len(xs_m):
            mask_pts = np.column_stack([xs_m, ys_m]).astype(np.float64)
            if len(mask_pts) > 400:
                mask_pts = np.ascontiguousarray(mask_pts[:: len(mask_pts) // 400])
            self._last_mask_pts = mask_pts
        else:
            self._last_mask_pts = None

    @staticmethod
    def _keypoint_reproj_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
        """RMS *pixel* error: project pitch points back into the image."""
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return float("inf")
        back = apply_homography(Hinv, dst)
        return float(np.sqrt(np.nanmean(np.sum((back - src) ** 2, axis=1))))
