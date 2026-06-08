"""
optimizer.py — Per-level L-BFGS optimisation loop for VFC + isoTV.

The objective at each pyramid level is::

    L(k) = vfc_loss_scale * VFC_align( fix , warp(mov, k) )
         + iso_tv         * TV_eps( k )
         [+ jac_reg       * mean(|det J - 1|)]   # (optional, simple form)

where ``k`` are control-point displacements in mm.
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple

import torch

from .regularizers import smooth_iso_tv
from .metrics_vfc import vfc_alignment_loss
from .warp import knots_to_dense, warp_image, forward_diff


def _knot_spacing_mm(
    grid_spacing: Tuple[float, float, float],
    pix_res: torch.Tensor,
) -> torch.Tensor:
    sp = torch.empty(3, device=pix_res.device, dtype=pix_res.dtype)
    for i, gs in enumerate(grid_spacing):
        sp[i] = float(gs) * pix_res[i]
    return sp


def _jac_penalty(
    knots_mm: torch.Tensor,
    knot_spacing_mm: torch.Tensor,
) -> torch.Tensor:
    """Cheap surrogate: penalise compression by mean (det J - 1)^2 on knots."""
    sp = knot_spacing_mm.to(knots_mm.device, dtype=knots_mm.dtype)
    J = torch.zeros((3, 3) + knots_mm.shape[1:],
                    device=knots_mm.device, dtype=knots_mm.dtype)
    for i in range(3):
        for j in range(3):
            J[i, j] = forward_diff(knots_mm[i], dim=j) / sp[j]
            if i == j:
                J[i, j] = J[i, j] + 1.0
    a, b, c = J[0, 0], J[0, 1], J[0, 2]
    d, e, f = J[1, 0], J[1, 1], J[1, 2]
    g, h, i_ = J[2, 0], J[2, 1], J[2, 2]
    detJ = a * (e * i_ - f * h) - b * (d * i_ - f * g) + c * (d * h - e * g)
    return (detJ - 1.0).pow(2).mean()


def optimise_level(
    knots_mm: torch.Tensor,
    fix_vfc: torch.Tensor,
    moving_image: torch.Tensor,
    edge_fn: Callable[[torch.Tensor], torch.Tensor],
    vfc_fn: Callable[[torch.Tensor], torch.Tensor],
    pix_res: torch.Tensor,
    grid_spacing: Tuple[float, float, float],
    iso_tv: float,
    csqrt: float,
    vfc_loss_scale: float,
    jac_reg: float,
    sign_invariant: bool,
    mask: Optional[torch.Tensor],
    max_iters: int,
    history: int,
    verbose: bool = False,
    level_tag: str = "",
) -> torch.Tensor:
    """
    Optimise ``knots_mm`` in place via L-BFGS at one pyramid level.

    Parameters
    ----------
    knots_mm     : (3, K1, K2, K3) initial knots in mm (will be cloned).
    fix_vfc      : (3, D, H, W) pre-computed VFC field of the fixed image.
    moving_image : (D, H, W) moving image at this level.
    edge_fn      : callable mapping a (D,H,W) image to a (D,H,W) edge map.
    vfc_fn       : callable mapping a (D,H,W) edge map to a (3,D,H,W) field.
    pix_res      : (3,) voxel spacing in mm at this level.
    grid_spacing : (3,) knot spacing in voxels at this level.
    iso_tv, csqrt, vfc_loss_scale, jac_reg, sign_invariant : objective weights.
    mask         : optional (D,H,W) weight map for the VFC term.
    max_iters    : L-BFGS iterations.
    history      : L-BFGS history size.
    """
    k = knots_mm.clone().detach().requires_grad_(True)
    img_shape = tuple(moving_image.shape)
    ksp = _knot_spacing_mm(grid_spacing, pix_res)

    opt = torch.optim.LBFGS(
        [k],
        lr=1.0,
        max_iter=max_iters,
        history_size=history,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-12,
    )

    state = {"it": 0, "best": float("inf")}

    def closure():
        opt.zero_grad()
        disp = knots_to_dense(k, img_shape)
        warped = warp_image(moving_image, disp, pix_res, mode="bilinear",
                            padding_mode="border")
        e_mov = edge_fn(warped)
        v_mov = vfc_fn(e_mov)

        loss_align = vfc_alignment_loss(
            fix_vfc, v_mov, mask=mask, sign_invariant=sign_invariant
        )
        loss_tv = smooth_iso_tv(k, ksp, csqrt=csqrt)
        loss = vfc_loss_scale * loss_align + iso_tv * loss_tv
        if jac_reg > 0.0:
            loss = loss + jac_reg * _jac_penalty(k, ksp)

        loss.backward()
        state["it"] += 1
        val = float(loss.detach())
        if val < state["best"]:
            state["best"] = val
        if verbose and (state["it"] == 1 or state["it"] % 5 == 0):
            print(f"  [{level_tag}] it={state['it']:3d}  "
                  f"L={val:.5f}  align={float(loss_align.detach()):.4f}  "
                  f"tv={float(loss_tv.detach()):.4f}")
        return loss

    opt.step(closure)
    return k.detach()
