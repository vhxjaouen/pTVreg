"""
warp.py — Control knots, dense displacement, and image warping.

Mirrors the conventions used in pyTVreg_v2/ptvreg_sonnet/warp.py so that
results and parameters transfer cleanly. Spatial layout is (D, H, W);
displacement field is (3, D, H, W) with component order (dD, dH, dW)
in millimetres.
"""

from __future__ import annotations
from typing import Tuple

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Knot grid sizing
# ---------------------------------------------------------------------------

def knot_grid_size(
    image_shape: Tuple[int, int, int],
    grid_spacing: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """Number of knots per axis: ceil(N_i / s_i) + 1 (linear B-spline)."""
    return tuple(int(math.ceil(n / s)) + 1 for n, s in zip(image_shape, grid_spacing))


# ---------------------------------------------------------------------------
# Forward finite difference (zero-Neumann at the far boundary)
# ---------------------------------------------------------------------------

def forward_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Forward difference along ``dim``. The last slice along ``dim`` is
    padded with zero so that the output has the same shape as ``x``.
    """
    s_fwd = [slice(None)] * x.ndim
    s_bwd = [slice(None)] * x.ndim
    s_fwd[dim] = slice(1, None)
    s_bwd[dim] = slice(None, -1)
    diff = x[tuple(s_fwd)] - x[tuple(s_bwd)]

    pad = [0] * (2 * x.ndim)
    rev = x.ndim - 1 - dim
    pad[2 * rev + 1] = 1
    return F.pad(diff, pad)


# ---------------------------------------------------------------------------
# Knots -> dense displacement
# ---------------------------------------------------------------------------

def knots_to_dense(
    knots_mm: torch.Tensor,
    image_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Trilinear interpolation from a control-point grid to image-sized grid.

    Parameters
    ----------
    knots_mm    : (3, K1, K2, K3) displacement knots in mm.
    image_shape : (D, H, W) target image size.

    Returns
    -------
    (3, D, H, W) dense displacement field in mm.
    """
    d = F.interpolate(
        knots_mm.unsqueeze(0),
        size=image_shape,
        mode="trilinear",
        align_corners=True,
    )
    return d.squeeze(0)


# ---------------------------------------------------------------------------
# Sampling grid for F.grid_sample (mm displacement -> normalised coords)
# ---------------------------------------------------------------------------

def _build_sampling_grid(
    disp_mm: torch.Tensor,
    pix_res: torch.Tensor,
) -> torch.Tensor:
    D, H, W = disp_mm.shape[1:]
    dev = disp_mm.device
    dtype = disp_mm.dtype
    pr = pix_res.to(dev, dtype=dtype)

    disp_pix = disp_mm / pr.view(3, 1, 1, 1)

    zz = torch.linspace(-1.0, 1.0, D, device=dev, dtype=dtype)
    yy = torch.linspace(-1.0, 1.0, H, device=dev, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, W, device=dev, dtype=dtype)
    gz, gy, gx = torch.meshgrid(zz, yy, xx, indexing="ij")

    dz = gz + disp_pix[0] * (2.0 / max(D - 1, 1))
    dy = gy + disp_pix[1] * (2.0 / max(H - 1, 1))
    dx = gx + disp_pix[2] * (2.0 / max(W - 1, 1))

    grid = torch.stack([dx, dy, dz], dim=-1)
    return grid.unsqueeze(0)


# ---------------------------------------------------------------------------
# Warping
# ---------------------------------------------------------------------------

def warp_image(
    image: torch.Tensor,
    disp_mm: torch.Tensor,
    pix_res: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """
    Warp a 3D volume by a dense displacement field, with the convention
    ``warped[x] = image[x + disp(x)]``.

    ``mode`` is passed through to F.grid_sample; use ``"nearest"`` for
    label maps.
    """
    grid = _build_sampling_grid(disp_mm, pix_res)
    out = F.grid_sample(
        image.unsqueeze(0).unsqueeze(0),
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )
    return out.squeeze(0).squeeze(0)


def warp_image_from_knots(
    image: torch.Tensor,
    knots_mm: torch.Tensor,
    pix_res: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    disp = knots_to_dense(knots_mm, image.shape)
    return warp_image(image, disp, pix_res, mode=mode, padding_mode=padding_mode)
