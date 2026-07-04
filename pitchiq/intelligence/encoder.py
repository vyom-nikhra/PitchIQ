"""Learned style encoder (6.1b): contrastive CNN over phase-conditioned heatmaps.

Input per player: a 3-channel image — overall / in-possession / defending
positional heatmaps in the attacking frame. Training uses the SimCLR-style
NT-Xent objective where the *same player in two different halves (or
matches)* forms a positive pair and all other players in the batch are
negatives: the encoder learns what is invariant about a player's movement
across contexts, i.e. their style.

``scripts/train_style_encoder.py`` trains on simulated matches out of the box
(and on any corpus of processed real matches via ``--jobs-dir``); torch is an
optional dependency (``pitchiq[cv]``) so everything here imports lazily.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pitchiq.config import EmbeddingsConfig
from pitchiq.intelligence.features import PlayerFeatures

log = logging.getLogger(__name__)


def build_encoder(dim: int = 64):
    """3xHxW heatmap stack → dim-d embedding (small CNN, ~80k params)."""
    import torch.nn as nn

    class StyleEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(96, dim),
            )

        def forward(self, x):
            import torch.nn.functional as F

            return F.normalize(self.net(x), dim=1)

    return StyleEncoder()


def heatmap_stack(pf: PlayerFeatures) -> np.ndarray:
    """(3, ny, nx) image: overall / attack / defend (missing → overall)."""
    base = pf.heatmap
    if base is None:
        raise ValueError(f"player {pf.entity_id} has no heatmap")
    att = pf.heatmap_attack if pf.heatmap_attack is not None else base
    dfd = pf.heatmap_defend if pf.heatmap_defend is not None else base
    stack = np.stack([base, att, dfd]).astype(np.float32)
    m = stack.max()
    return stack / m if m > 0 else stack


def nt_xent_loss(z1, z2, temperature: float = 0.2):
    """SimCLR NT-Xent over a batch of positive pairs (z1[i], z2[i])."""
    import torch
    import torch.nn.functional as F

    z = torch.cat([z1, z2], dim=0)              # (2B, D), already normalised
    sim = z @ z.t() / temperature               # (2B, 2B)
    B = z1.shape[0]
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


def train_style_encoder(pair_sets: list[tuple[np.ndarray, np.ndarray]],
                        dim: int = 64, epochs: int = 60, lr: float = 2e-3,
                        batch_size: int = 32, seed: int = 0,
                        out_path: str | Path = "weights/style_encoder.pt") -> dict:
    """Train on positive pairs of heatmap stacks; saves torchscript weights.

    ``pair_sets``: list of ((3,ny,nx), (3,ny,nx)) same-player different-context
    images. Returns training diagnostics.
    """
    import torch

    torch.manual_seed(seed)
    model = build_encoder(dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    A = torch.from_numpy(np.stack([p[0] for p in pair_sets]))
    B = torch.from_numpy(np.stack([p[1] for p in pair_sets]))
    n = len(pair_sets)
    losses = []
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            if len(idx) < 4:
                continue
            a, b = A[idx], B[idx]
            # light augmentation: lateral flip (y symmetry) applied jointly
            if torch.rand(1).item() < 0.5:
                a = torch.flip(a, dims=[2])
                b = torch.flip(b, dims=[2])
            loss = nt_xent_loss(model(a), model(b))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
        losses.append(ep_loss)
    model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    example = A[:2]
    torch.jit.trace(model, example).save(str(out_path))
    return {"epochs": epochs, "final_loss": losses[-1], "n_pairs": n,
            "weights": str(out_path)}


def embed_with_encoder(features: dict[int, PlayerFeatures],
                       cfg: EmbeddingsConfig):
    """Inference path used by :func:`compute_embeddings`; None if unusable."""
    from pitchiq.intelligence.embeddings import EmbeddingResult, robust_scale

    weights = cfg.learned.weights
    if not weights or not Path(weights).exists():
        return None
    import torch

    model = torch.jit.load(weights, map_location="cpu").eval()
    ids = sorted(features)
    stacks = np.stack([heatmap_stack(features[i]) for i in ids])
    with torch.no_grad():
        V = model(torch.from_numpy(stacks)).numpy()
    # keep scalar matrix for attribution even in learned mode
    flat0, names = features[ids[0]].flat()
    scal = np.stack([features[i].flat()[0] for i in ids])
    from pitchiq.intelligence.embeddings import _group_slices

    log.info("learned style encoder used: %s", weights)
    return EmbeddingResult(ids=ids, vectors=V.astype(np.float32), method="learned",
                           feature_names=names, scalar_matrix=robust_scale(scal),
                           group_slices=_group_slices(names))
