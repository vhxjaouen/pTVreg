"""
vfc.py — Vector Field Convolution on PyTorch tensors (GPU-friendly).

Implements the VFC kernel from Li & Acton (2007) and applies it via FFT
convolution to a scalar edge map, producing a smooth vector field that
points inward toward edges (i.e. toward the kernel center).

Conventions
-----------
- Spatial layout throughout the package: (D, H, W).
- Vector field component order: (vD, vH, vW), matching the spatial axes
  in the same (D, H, W) order.
- All tensors are differentiable w.r.t. their inputs when invoked with
  PyTorch autograd; the kernel itself is precomputed and not learned.

Notes
-----
- The kernel center contributes zero force (no singularity).
- Kernel radius is specified in physical units (mm) and converted to
  voxels using `pix_res`, so the operator is voxel-spacing aware.
- 2D input (H, W) is auto-promoted to (1, H, W) internally and the
  output is squeezed back if requested.
"""

from __future__ import annotations
from typing import Optional, Tuple

import math
import torch


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def build_vfc_kernel(
    radius_mm: float,
    pix_res: torch.Tensor,
    gamma: float = 1.7,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build a 3D inward-pointing VFC kernel.

    Parameters
    ----------
    radius_mm : kernel radius in millimetres.
    pix_res   : (3,) voxel spacing in mm, in (D, H, W) order.
    gamma     : decay exponent. Magnitude ~ 1 / r^(gamma+1).
                Larger gamma -> more localized (NGF-like).
                Smaller gamma -> smoother, longer range.
    device    : torch device for the kernel tensor.
    dtype     : kernel dtype.
    eps       : small constant to avoid division by zero at the centre.

    Returns
    -------
    (3, Kd, Kh, Kw) tensor whose vectors point inward toward the centre,
    with component order (vD, vH, vW). The centre voxel is exactly zero.
    """
    if pix_res.numel() != 3:
        raise ValueError("pix_res must be a length-3 tensor (D, H, W spacing).")
    pix_res = pix_res.to(device=device, dtype=dtype)

    # Effective radius: must cover at least the 6-connected neighbourhood,
    # otherwise the radial mask drops every off-centre voxel and the kernel
    # collapses to zero (typical at coarse pyramid levels where the voxel
    # spacing in mm can exceed the user-requested radius_mm).
    max_pix = float(pix_res.max().item())
    eff_radius = float(radius_mm)
    if eff_radius < 1.01 * max_pix:
        import warnings
        warnings.warn(
            f"VFC radius {eff_radius:.3g} mm is smaller than the voxel "
            f"spacing ({max_pix:.3g} mm); clamping kernel support to "
            f"{1.01 * max_pix:.3g} mm so it spans at least the 6-neighbours.",
            stacklevel=2,
        )
        eff_radius = 1.01 * max_pix

    # Radius along each axis in voxels (>= 1 voxel each side)
    r_vox = [max(1, int(math.ceil(eff_radius / float(pix_res[i].item()))))
             for i in range(3)]

    # Coordinate grids in mm, centred at zero
    def _axis(n: int, sp: float) -> torch.Tensor:
        coords = torch.arange(-n, n + 1, device=device, dtype=dtype)
        return coords * sp

    zd = _axis(r_vox[0], float(pix_res[0]))
    zh = _axis(r_vox[1], float(pix_res[1]))
    zw = _axis(r_vox[2], float(pix_res[2]))

    gd, gh, gw = torch.meshgrid(zd, zh, zw, indexing="ij")   # each (Kd,Kh,Kw)
    r = torch.sqrt(gd * gd + gh * gh + gw * gw)              # mm

    # Spherical support: r <= eff_radius
    mask = (r <= eff_radius).to(dtype)

    # Magnitude: 1 / r^(gamma+1), zero at centre
    denom = r.pow(gamma + 1.0) + eps
    mag = mask / denom
    # explicit zero at centre to avoid any residual 1/eps spike
    centre = (r == 0)
    mag = mag.masked_fill(centre, 0.0)

    # Inward direction = -coordinate / r
    safe_r = r.clamp(min=eps)
    kD = -gd / safe_r * mag
    kH = -gh / safe_r * mag
    kW = -gw / safe_r * mag

    kernel = torch.stack([kD, kH, kW], dim=0)   # (3, Kd, Kh, Kw)
    # Normalize by peak per-component magnitude so the VFC field is on the
    # same order as the edge map. Without this, the gradient of the cosine
    # alignment loss w.r.t. displacement is O(1e-5) and L-BFGS strong-Wolfe
    # cannot take a meaningful step at coarse pyramid levels.
    peak = kernel.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=eps)
    kernel = kernel / peak
    return kernel


# ---------------------------------------------------------------------------
# FFT-based 3D convolution
# ---------------------------------------------------------------------------

def _next_fft_size(n: int) -> int:
    """Round n up to the next power of two for FFT efficiency."""
    return 1 << (max(1, n - 1)).bit_length()


def _fft_conv3d_same(
    volume: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """
    Linear 3D convolution of a single-channel volume with a single
    real kernel, returning the central 'same'-sized result.

    Implements zero-padded linear convolution via FFT, so it is
    boundary-safe (no circular wrap-around).

    Parameters
    ----------
    volume : (D, H, W) real tensor.
    kernel : (Kd, Kh, Kw) real tensor.

    Returns
    -------
    (D, H, W) real tensor.
    """
    if volume.ndim != 3 or kernel.ndim != 3:
        raise ValueError("volume and kernel must both be 3D.")
    D, H, W = volume.shape
    Kd, Kh, Kw = kernel.shape

    fft_shape = (
        _next_fft_size(D + Kd - 1),
        _next_fft_size(H + Kh - 1),
        _next_fft_size(W + Kw - 1),
    )

    Fv = torch.fft.rfftn(volume, s=fft_shape)
    Fk = torch.fft.rfftn(kernel, s=fft_shape)
    conv = torch.fft.irfftn(Fv * Fk, s=fft_shape)

    # Centre crop to (D, H, W) — kernel-centred 'same' convolution
    sd = Kd // 2
    sh = Kh // 2
    sw = Kw // 2
    return conv[sd:sd + D, sh:sh + H, sw:sw + W]


# ---------------------------------------------------------------------------
# Public API: scalar edge map -> VFC vector field
# ---------------------------------------------------------------------------

def vfc_field(
    edge_map: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """
    Apply VFC to a scalar edge map.

    Parameters
    ----------
    edge_map : (D, H, W) real tensor, typically a non-negative edge
               magnitude (e.g. |∇I|² or label-boundary magnitude).
    kernel   : (3, Kd, Kh, Kw) VFC kernel from :func:`build_vfc_kernel`.

    Returns
    -------
    (3, D, H, W) vector field, component order (vD, vH, vW).
    """
    if edge_map.ndim != 3:
        raise ValueError(f"edge_map must be (D,H,W), got {tuple(edge_map.shape)}")
    if kernel.ndim != 4 or kernel.shape[0] != 3:
        raise ValueError("kernel must be (3, Kd, Kh, Kw).")

    # FFT once on the volume, multiply by each component kernel.
    D, H, W = edge_map.shape
    Kd, Kh, Kw = kernel.shape[1:]
    fft_shape = (
        _next_fft_size(D + Kd - 1),
        _next_fft_size(H + Kh - 1),
        _next_fft_size(W + Kw - 1),
    )
    Fv = torch.fft.rfftn(edge_map, s=fft_shape)

    out = torch.empty(3, D, H, W, device=edge_map.device, dtype=edge_map.dtype)
    sd = Kd // 2
    sh = Kh // 2
    sw = Kw // 2
    for c in range(3):
        Fk = torch.fft.rfftn(kernel[c], s=fft_shape)
        conv = torch.fft.irfftn(Fv * Fk, s=fft_shape)
        out[c] = conv[sd:sd + D, sh:sh + H, sw:sw + W]
    return out


def normalize_vfc(
    field: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize a vector field per-voxel with a smooth (differentiable) norm.

    The magnitude is computed as ``sqrt(sum(field^2) + eps^2)`` so its
    gradient stays bounded even where the field is exactly zero — this
    matters because plain ``sqrt(0)`` has an infinite derivative and
    will poison L-BFGS with NaNs after the first step.

    Parameters
    ----------
    field : (3, D, H, W) vector field.
    eps   : floor on magnitude (same units as the field).

    Returns
    -------
    unit_field : (3, D, H, W) near-unit vectors (true unit where the
                 underlying magnitude is much larger than ``eps``).
    magnitude  : (D, H, W) smoothed magnitudes (always > 0).
    """
    mag2 = field.pow(2).sum(dim=0)
    mag = (mag2 + eps * eps).sqrt()
    unit = field / mag.unsqueeze(0)
    return unit, mag
