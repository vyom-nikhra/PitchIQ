"""Tracking-quality metrics: MOTA, IDF1, ID switches.

Self-contained implementations of the CLEAR-MOT and identity metrics
(Bernardin & Stiefelhagen 2008; Ristani et al. 2016) so we can score the
tracker against SoccerNet MOT ground truth or the synthetic simulator's
ground-truth tracks without a motmetrics dependency.

Both ``gt`` and ``pred`` are dataframes with columns
``frame, track_id, x1, y1, x2, y2``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from pitchiq.core.types import iou_matrix


def evaluate_tracking(gt: pd.DataFrame, pred: pd.DataFrame, iou_thresh: float = 0.5) -> dict:
    """Compute MOTA, IDF1, ID-switch count and supporting tallies."""
    frames = sorted(set(gt["frame"]).union(pred["frame"]))
    gt_by_frame = {f: g for f, g in gt.groupby("frame")}
    pr_by_frame = {f: p for f, p in pred.groupby("frame")}

    fp = fn = idsw = matches_total = 0
    gt_total = len(gt)
    last_match: dict[int, int] = {}  # gt_id -> pred_id from previous frames
    # accumulate co-occurrence for IDF1
    overlap_counts: dict[tuple[int, int], int] = {}
    gt_id_counts: dict[int, int] = {}
    pr_id_counts: dict[int, int] = {}

    for f in frames:
        g = gt_by_frame.get(f)
        p = pr_by_frame.get(f)
        g_ids = g["track_id"].to_numpy() if g is not None else np.array([], dtype=int)
        p_ids = p["track_id"].to_numpy() if p is not None else np.array([], dtype=int)
        for gid in g_ids:
            gt_id_counts[gid] = gt_id_counts.get(gid, 0) + 1
        for pid in p_ids:
            pr_id_counts[pid] = pr_id_counts.get(pid, 0) + 1

        if g is None or len(g) == 0:
            fp += 0 if p is None else len(p)
            continue
        if p is None or len(p) == 0:
            fn += len(g)
            continue

        g_boxes = g[["x1", "y1", "x2", "y2"]].to_numpy()
        p_boxes = p[["x1", "y1", "x2", "y2"]].to_numpy()
        iou = iou_matrix(g_boxes, p_boxes)

        # CLEAR-MOT: prefer persisting the previous match (hysteresis)
        cost = 1.0 - iou
        for gi, gid in enumerate(g_ids):
            prev = last_match.get(gid)
            if prev is not None:
                pj = np.where(p_ids == prev)[0]
                if len(pj) and iou[gi, pj[0]] >= iou_thresh:
                    cost[gi, pj[0]] -= 0.2  # bias toward continuity
        rows, cols = linear_sum_assignment(cost)
        matched_g, matched_p = set(), set()
        for r, c in zip(rows, cols):
            if iou[r, c] < iou_thresh:
                continue
            gid, pid = int(g_ids[r]), int(p_ids[c])
            matched_g.add(r)
            matched_p.add(c)
            matches_total += 1
            overlap_counts[(gid, pid)] = overlap_counts.get((gid, pid), 0) + 1
            if gid in last_match and last_match[gid] != pid:
                idsw += 1
            last_match[gid] = pid
        fn += len(g_ids) - len(matched_g)
        fp += len(p_ids) - len(matched_p)

    mota = 1.0 - (fn + fp + idsw) / max(gt_total, 1)

    # IDF1: optimal global gt-id <-> pred-id bijection maximising co-occurrence
    gt_uids = sorted(gt_id_counts)
    pr_uids = sorted(pr_id_counts)
    if gt_uids and pr_uids:
        w = np.zeros((len(gt_uids), len(pr_uids)))
        for (gid, pid), c in overlap_counts.items():
            w[gt_uids.index(gid), pr_uids.index(pid)] = c
        rr, cc = linear_sum_assignment(-w)
        idtp = w[rr, cc].sum()
    else:
        idtp = 0.0
    idf1 = 2.0 * idtp / max(gt_total + len(pred), 1)

    return {
        "mota": round(float(mota), 4),
        "idf1": round(float(idf1), 4),
        "id_switches": int(idsw),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "matches": int(matches_total),
        "gt_boxes": int(gt_total),
        "pred_boxes": int(len(pred)),
    }
