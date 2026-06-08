"""
pyramid.py — Gaussian image pyramid (coarsest -> finest).

Lightweight reimplementation of the pyramid used in pyTVreg_v2 so that
this package stays self-contained.
"""

from __future__ import annotations
from typing import List, Tuple

import math
import torch
import torch.nn.functional as F


def auto_nlvl(
    image_shape: Tuple[int, int, int],
    grid_spacing: Tuple[float, float, float],
    k_down: float = 0.7,
) -> int:
    """Pyramid level count s.t. the coarsest image still fits at least
    a few knots on every axis."""
    log_scale = math.log(1.0 / k_down)
    levels = []
    for n, s in zip(image_shape, grid_spacing):
        denom = max(4.0 * float(s), 16.0)
        if n <= denom:
            levels.append(1)
        else:
            levels.append(int(math.floor(math.log(n / denom) / log_scale)))
    return max(1, min(levels))


def _gaussian_blur3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma < 1e-6:
        return x
    radius = max(1, int(math.ceil(3.0 * sigma)))
    ks = 2 * radius + 1
    coords = torch.arange(ks, dtype=x.dtype, device=x.device) - radius
    k1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    k1d = k1d / k1d.sum()

    vol = x.unsqueeze(0).unsqueeze(0)
    pad = radius
    k = k1d.view(1, 1, ks, 1, 1)
    vol = F.conv3d(vol, k, padding=(pad, 0, 0))
    k = k1d.view(1, 1, 1, ks, 1)
    vol = F.conv3d(vol, k, padding=(0, pad, 0))
    k = k1d.view(1, 1, 1, 1, ks)
    vol = F.conv3d(vol, k, padding=(0, 0, pad))
    return vol.squeeze(0).squeeze(0)


def build_pyramid(
    image: torch.Tensor,
    nlvl: int,
    pix_res: torch.Tensor,
    k_down: float = 0.7,
    mode: str = "trilinear",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Build a Gaussian pyramid for a single 3D volume.

    Parameters
    ----------
    image   : (D, H, W) float tensor.
    nlvl    : number of pyramid levels (1 -> only the original).
    pix_res : (3,) voxel spacing at the original (finest) scale, in mm.
    k_down  : per-level downsampling factor (< 1).
    mode    : interpolation mode for downsampling. Use ``"nearest"`` for
              label maps; default is ``"trilinear"`` for intensity images.

    Returns
    -------
    List of (img, pix_res) tuples, coarsest first, finest last.
    """
    pyramid: List[Tuple[torch.Tensor, torch.Tensor]] = [(image, pix_res)]
    sigma_anti = 0.4 * 0.5 / k_down

    current = image
    current_pix = pix_res
    for _ in range(nlvl - 1):
        if mode == "nearest":
            blurred = current   # do not blur label maps
        else:
            blurred = _gaussian_blur3d(current, sigma_anti)
        new_size = tuple(max(2, int(math.ceil(n * k_down))) for n in blurred.shape)
        kw = {"align_corners": True} if mode != "nearest" else {}
        downsampled = F.interpolate(
            blurred.unsqueeze(0).unsqueeze(0),
            size=new_size,
            mode=mode,
            **kw,
        ).squeeze()
        new_pix = current_pix / k_down
        pyramid.append((downsampled, new_pix))
        current = downsampled
        current_pix = new_pix

    pyramid.reverse()
    return pyramid
