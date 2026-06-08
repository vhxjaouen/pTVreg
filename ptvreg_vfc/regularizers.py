"""
regularizers.py — Smoothed isotropic Total Variation on control knots.

Identical in spirit to pyTVreg_v2/ptvreg_sonnet/regularizers.py:

    TV_eps(k) = mean_l sqrt( sum_{i,j} (dk_i / dx_j [l])^2 + eps )

with physical knot-spacing normalization. PyTorch autograd handles the
gradient end-to-end.
"""

from __future__ import annotations

import torch

from .warp import forward_diff


def smooth_iso_tv(
    knots_mm: torch.Tensor,
    knot_spacing_mm: torch.Tensor,
    csqrt: float = 1e-2,
) -> torch.Tensor:
    """
    Parameters
    ----------
    knots_mm        : (3, K1, K2, K3) control knot displacements in mm.
    knot_spacing_mm : (3,) physical distance between adjacent knots in mm,
                      i.e. ``pix_res * grid_spacing`` element-wise.
    csqrt           : smoothing constant epsilon.

    Returns
    -------
    Scalar TV value (mean over knot positions).
    """
    if knots_mm.ndim != 4 or knots_mm.shape[0] != 3:
        raise ValueError("knots_mm must have shape (3, K1, K2, K3).")
    if knot_spacing_mm.numel() != 3:
        raise ValueError("knot_spacing_mm must be length 3.")

    sp = knot_spacing_mm.to(knots_mm.device, dtype=knots_mm.dtype)
    sq_sum = torch.zeros_like(knots_mm[0])
    for i in range(3):
        for j in range(3):
            dk = forward_diff(knots_mm[i], dim=j) / sp[j]
            sq_sum = sq_sum + dk.pow(2)
    return (sq_sum + csqrt).sqrt().mean()
