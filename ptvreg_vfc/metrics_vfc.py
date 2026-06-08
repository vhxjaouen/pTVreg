"""
metrics_vfc.py — VFC-based similarity loss.

The fixed-image VFC field is computed once per pyramid level.
The moving-image VFC field is recomputed every iteration from the
*warped* edge map and aligned against the fixed VFC via cosine
similarity (or its sign-invariant variant).
"""

from __future__ import annotations
from typing import Optional

import torch

from .vfc import vfc_field, normalize_vfc


def vfc_alignment_loss(
    vfc_fix: torch.Tensor,
    vfc_mov: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    sign_invariant: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Negative mean cosine alignment between two VFC vector fields.

    Parameters
    ----------
    vfc_fix, vfc_mov : (3, D, H, W) vector fields.
    mask             : optional (D, H, W) non-negative weight map.
    sign_invariant   : if True use cos^2 (good for label maps where the
                       boundary normal can flip sign in flat patches).
    """
    if vfc_fix.shape != vfc_mov.shape:
        raise ValueError(
            f"VFC shape mismatch: fix {tuple(vfc_fix.shape)} mov {tuple(vfc_mov.shape)}"
        )

    n_fix, _ = normalize_vfc(vfc_fix, eps=eps)
    n_mov, _ = normalize_vfc(vfc_mov, eps=eps)
    dot = (n_fix * n_mov).sum(dim=0)
    if sign_invariant:
        dot = dot.pow(2)

    if mask is None:
        return -dot.mean()
    w = mask.to(dot.dtype).clamp(min=0.0)
    denom = w.sum().clamp(min=eps)
    return -(dot * w).sum() / denom


def vfc_field_from_edges(
    edge_map: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Thin wrapper kept for API symmetry."""
    return vfc_field(edge_map, kernel)
