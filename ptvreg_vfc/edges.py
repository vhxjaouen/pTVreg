"""
edges.py — Edge-map extraction for VFC.

Two sources are supported:

* ``image``  — gradient-magnitude edge map of an intensity image.
* ``labels`` — boundary map of a multi-label segmentation. Label IDs
  do NOT influence vector magnitude: we gradient each one-hot channel
  and accumulate the magnitude, optionally per-label-weighted.

The resulting edge map is a non-negative scalar (D, H, W) tensor, ready
to be fed into :func:`ptvreg_vfc.vfc.vfc_field`.

All operations are differentiable in PyTorch and run on GPU when the
input tensors do.
"""

from __future__ import annotations
from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient (central differences with zero padding, mm-aware)
# ---------------------------------------------------------------------------

def _central_grad(volume: torch.Tensor, pix_res: torch.Tensor) -> torch.Tensor:
    """
    Central-difference gradient of a (D, H, W) volume.
    Returns (3, D, H, W) gradient in units of [intensity / mm].
    Boundaries use zero-padded one-sided differences.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D, got {tuple(volume.shape)}")
    if pix_res.numel() != 3:
        raise ValueError("pix_res must be length 3.")
    pr = pix_res.to(volume.device, dtype=volume.dtype)

    out = []
    for d in range(3):
        s_fwd = [slice(None)] * 3
        s_bwd = [slice(None)] * 3
        s_fwd[d] = slice(2, None)
        s_bwd[d] = slice(None, -2)
        diff = volume[tuple(s_fwd)] - volume[tuple(s_bwd)]   # shape[d] = N - 2

        pad = [0] * 6
        rev = 2 - d
        pad[2 * rev] = 1
        pad[2 * rev + 1] = 1
        diff = F.pad(diff, pad) / (2.0 * pr[d])
        out.append(diff)
    return torch.stack(out, dim=0)


# ---------------------------------------------------------------------------
# Public: image edge map
# ---------------------------------------------------------------------------

def image_edge_map(
    image: torch.Tensor,
    pix_res: torch.Tensor,
    use_kornia: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Edge magnitude map of an intensity image.

    Parameters
    ----------
    image      : (D, H, W) float tensor.
    pix_res    : (3,) voxel spacing in mm.
    use_kornia : if True, try to use ``kornia.filters.spatial_gradient3d``
                 for the gradient (otherwise central differences are used).
    eps        : small floor inside the square root.

    Returns
    -------
    (D, H, W) non-negative edge magnitude in units of [intensity / mm].
    """
    if use_kornia:
        try:
            from kornia.filters import SpatialGradient3d
            grad = SpatialGradient3d()(image[None, None])  # (1,1,3,D,H,W)
            grad = grad[0, 0]                              # (3,D,H,W)
            # Kornia gradients are in voxel units -> divide by spacing
            pr = pix_res.to(image.device, dtype=image.dtype).view(3, 1, 1, 1)
            grad = grad / pr
        except Exception:
            grad = _central_grad(image, pix_res)
    else:
        grad = _central_grad(image, pix_res)

    return (grad.pow(2).sum(dim=0) + eps).sqrt()


# ---------------------------------------------------------------------------
# Public: label edge map
# ---------------------------------------------------------------------------

def label_edge_map(
    labels: torch.Tensor,
    pix_res: torch.Tensor,
    label_ids: Optional[Sequence[int]] = None,
    per_label_weights: Optional[Sequence[float]] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Boundary magnitude map from a multi-label volume.

    The output is the L2-norm of the per-label gradient stack, optionally
    weighted, so label *identity* does not affect vector magnitude — only
    *boundaries* do.

    Parameters
    ----------
    labels             : (D, H, W) integer or float tensor of label IDs.
                         Background should normally be 0.
    pix_res            : (3,) voxel spacing in mm.
    label_ids          : explicit list of label IDs to use. If None, all
                         unique non-zero values found in ``labels`` are used.
    per_label_weights  : optional per-label weights aligned with ``label_ids``.
                         Defaults to 1.0 for each label.
    eps                : small floor inside the square root.

    Returns
    -------
    (D, H, W) non-negative boundary magnitude.
    """
    if labels.ndim != 3:
        raise ValueError(f"labels must be 3D, got {tuple(labels.shape)}")

    if label_ids is None:
        unique = torch.unique(labels)
        ids = [int(x.item()) for x in unique if int(x.item()) != 0]
    else:
        ids = [int(i) for i in label_ids]

    if not ids:
        return torch.zeros_like(labels, dtype=torch.float32)

    if per_label_weights is None:
        weights = [1.0] * len(ids)
    else:
        if len(per_label_weights) != len(ids):
            raise ValueError(
                f"per_label_weights length {len(per_label_weights)} "
                f"does not match label_ids length {len(ids)}"
            )
        weights = [float(w) for w in per_label_weights]

    dtype = torch.float32 if not labels.is_floating_point() else labels.dtype
    accum = torch.zeros(labels.shape, device=labels.device, dtype=dtype)

    for lid, w in zip(ids, weights):
        mask = (labels == lid).to(dtype)
        grad = _central_grad(mask, pix_res)         # (3,D,H,W)
        accum = accum + (w * w) * grad.pow(2).sum(dim=0)

    return (accum + eps).sqrt()


# ---------------------------------------------------------------------------
# Public: hybrid edge map
# ---------------------------------------------------------------------------

def hybrid_edge_map(
    image_edges: torch.Tensor,
    label_edges: torch.Tensor,
    weight: float = 0.5,
) -> torch.Tensor:
    """
    Convex combination of image-derived and label-derived edge maps.

    Each input is rescaled to [0, 1] by its own 99th-percentile before
    blending, so the two sources are commensurable regardless of their
    raw units.

    Parameters
    ----------
    image_edges : (D, H, W) non-negative.
    label_edges : (D, H, W) non-negative.
    weight      : in [0, 1]. ``weight=1`` => labels only, ``weight=0`` =>
                  image only.
    """
    if image_edges.shape != label_edges.shape:
        raise ValueError(
            f"shape mismatch: image {tuple(image_edges.shape)} vs "
            f"labels {tuple(label_edges.shape)}"
        )
    w = float(weight)
    w = max(0.0, min(1.0, w))

    def _norm(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten()
        n = flat.numel()
        if n <= 16_000_000:
            q = torch.quantile(flat, 0.99)
        else:
            k = max(1, min(n, int(round(0.99 * (n - 1))) + 1))
            q = torch.kthvalue(flat, k).values
        scale = q.clamp(min=1e-8)
        return (x / scale).clamp(min=0.0)

    return (1.0 - w) * _norm(image_edges) + w * _norm(label_edges)
