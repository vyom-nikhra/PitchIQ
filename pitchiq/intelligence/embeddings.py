"""Player-style embeddings: handcrafted (6.1a) and learned (6.1b).

**Handcrafted (always available).** Scalar feature groups are z-scored over
the player population (robust: median/IQR), the downsampled heatmap is
appended with a global weight, and PCA reduces to ``pca_dim``. Interpretable:
group slices are preserved so similarity can be attributed to movement vs
involvement vs territory.

**Learned (optional).** A small CNN encodes the 3-channel phase-conditioned
heatmap image (overall / in-possession / defending) trained with a SimCLR-
style contrastive objective where the same player in different halves (or
different matches) is a positive pair (:mod:`pitchiq.intelligence.encoder`,
``scripts/train_style_encoder.py``). If torch or trained weights are absent
the system transparently uses the handcrafted embedding — recorded in the
artifact so results are traceable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from pitchiq.config import EmbeddingsConfig
from pitchiq.intelligence.features import PlayerFeatures

log = logging.getLogger(__name__)

HEATMAP_WEIGHT = 0.6  # relative weight of heatmap block vs scalar features


@dataclass
class EmbeddingResult:
    ids: list[int]
    vectors: np.ndarray            # (N, D) L2-normalised
    method: str                    # 'handcrafted' | 'learned'
    feature_names: list[str]       # scalar feature names (pre-PCA)
    scalar_matrix: np.ndarray      # (N, F) robust-scaled scalars (for attribution)
    group_slices: dict[str, slice]


def robust_scale(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    iqr = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
    iqr[iqr < 1e-9] = 1.0
    Z = (X - med) / iqr
    return np.clip(np.nan_to_num(Z), -4, 4)


def compute_handcrafted(features: dict[int, PlayerFeatures],
                        cfg: EmbeddingsConfig) -> EmbeddingResult:
    ids = sorted(features)
    flat0, names = features[ids[0]].flat()
    scalars = np.zeros((len(ids), len(names)))
    heatmaps = []
    for i, eid in enumerate(ids):
        vec, nm = features[eid].flat()
        if nm != names:  # align by name if group availability differs
            aligned = np.zeros(len(names))
            lookup = dict(zip(nm, vec))
            for j, n in enumerate(names):
                aligned[j] = lookup.get(n, 0.0)
            vec = aligned
        scalars[i] = vec
        hm = features[eid].heatmap
        heatmaps.append(hm.ravel() if hm is not None else np.zeros(cfg.heatmap_nx * cfg.heatmap_ny))

    Z = robust_scale(scalars)
    H = np.stack(heatmaps)
    # scale heatmap block to comparable magnitude
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
    X = np.hstack([Z, HEATMAP_WEIGHT * np.sqrt(Z.shape[1]) * H])

    # PCA (skip if too few players for a stable projection)
    dim = min(cfg.pca_dim, len(ids) - 1, X.shape[1])
    if dim >= 4 and len(ids) >= 8:
        from sklearn.decomposition import PCA

        V = PCA(n_components=dim, random_state=0).fit_transform(X)
    else:
        V = X
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    group_slices = _group_slices(names)
    return EmbeddingResult(ids=ids, vectors=V.astype(np.float32), method="handcrafted",
                           feature_names=names, scalar_matrix=Z, group_slices=group_slices)


def _group_slices(names: list[str]) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    start = 0
    cur = names[0].split(".")[0]
    for i, n in enumerate(names):
        g = n.split(".")[0]
        if g != cur:
            slices[cur] = slice(start, i)
            start, cur = i, g
    slices[cur] = slice(start, len(names))
    return slices


def compute_embeddings(features: dict[int, PlayerFeatures],
                       cfg: EmbeddingsConfig) -> EmbeddingResult:
    """Learned encoder when enabled+available, else handcrafted (logged)."""
    use_learned = cfg.learned.enabled is True or (
        cfg.learned.enabled == "auto" and cfg.learned.weights)
    if use_learned:
        try:
            from pitchiq.intelligence.encoder import embed_with_encoder

            result = embed_with_encoder(features, cfg)
            if result is not None:
                return result
        except Exception as exc:
            log.warning("learned embedding unavailable (%s); using handcrafted", exc)
    return compute_handcrafted(features, cfg)
