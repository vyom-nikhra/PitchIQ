"""Similar-player search: cosine kNN over style embeddings + attribution.

FAISS (IndexFlatIP over L2-normalised vectors = exact cosine) when installed;
transparently falls back to sklearn NearestNeighbors. Each hit is explained
by per-feature-group cosine similarity on the scaled scalar features —
"similar because of movement profile, not territory".
"""

from __future__ import annotations

import logging

import numpy as np

from pitchiq.config import SimilarityConfig
from pitchiq.intelligence.embeddings import EmbeddingResult

log = logging.getLogger(__name__)


class SimilarityIndex:
    def __init__(self, emb: EmbeddingResult, cfg: SimilarityConfig) -> None:
        self.emb = emb
        self.cfg = cfg
        self.ids = emb.ids
        V = np.ascontiguousarray(emb.vectors.astype(np.float32))
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        self.V = V
        self.backend = "sklearn"
        self._faiss = None
        if cfg.backend in ("auto", "faiss"):
            try:
                import faiss

                index = faiss.IndexFlatIP(V.shape[1])
                index.add(V)
                self._faiss = index
                self.backend = "faiss"
            except Exception as exc:  # pragma: no cover — env dependent
                if cfg.backend == "faiss":
                    log.warning("faiss requested but unavailable: %s", exc)
        if self._faiss is None:
            from sklearn.neighbors import NearestNeighbors

            self._nn = NearestNeighbors(metric="cosine").fit(V)
        log.info("similarity index backend: %s (%d players)", self.backend, len(self.ids))

    def query(self, entity_id: int, k: int | None = None) -> list[dict]:
        """Top-k most similar players (excluding self) with attributions."""
        k = k or self.cfg.top_k
        if entity_id not in self.ids:
            raise KeyError(f"unknown entity {entity_id}")
        qi = self.ids.index(entity_id)
        q = self.V[qi: qi + 1]
        kk = min(k + 1, len(self.ids))
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, kk)
            pairs = [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        else:
            dist, idx = self._nn.kneighbors(q, n_neighbors=kk)
            pairs = [(int(i), 1.0 - float(d)) for i, d in zip(idx[0], dist[0])]
        out = []
        for i, score in pairs:
            if self.ids[i] == entity_id:
                continue
            out.append({
                "entity_id": self.ids[i],
                "similarity": round(score, 4),
                "drivers": self.explain(qi, i),
            })
        return out[:k]

    def explain(self, i: int, j: int) -> dict[str, float]:
        """Per-feature-group cosine similarity between players i and j."""
        out = {}
        Z = self.emb.scalar_matrix
        for group, sl in self.emb.group_slices.items():
            a, b = Z[i, sl], Z[j, sl]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            out[group] = round(float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0, 3)
        return out

    def all_neighbors(self, k: int | None = None) -> dict[str, list[dict]]:
        return {str(eid): self.query(eid, k) for eid in self.ids}
