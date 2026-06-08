"""
io.py — Minimal NIfTI I/O for the VFC registration package.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch

try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError(
        "nibabel is required. Install with: pip install nibabel"
    ) from exc


def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a NIfTI file.

    Returns
    -------
    data     : (D, H, W) float32 numpy array in PyTorch axis order.
    affine   : (4, 4) float64 voxel-to-world affine.
    voxel_mm : (3,) float32 voxel sizes in mm in (D, H, W) order.
    """
    img = nib.load(path)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    data = data.transpose(2, 1, 0)
    affine = np.array(img.affine, dtype=np.float64)
    voxel_mm = np.array(img.header.get_zooms()[:3], dtype=np.float32)[::-1].copy()
    return data, affine, voxel_mm


def load_label_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like :func:`load_nifti` but keeps integer label values."""
    img = nib.load(path)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    data = np.rint(data).astype(np.int32)
    data = data.transpose(2, 1, 0)
    affine = np.array(img.affine, dtype=np.float64)
    voxel_mm = np.array(img.header.get_zooms()[:3], dtype=np.float32)[::-1].copy()
    return data, affine, voxel_mm


def save_nifti(
    data,
    path: str,
    ref_affine: np.ndarray,
) -> None:
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = np.asarray(data).astype(np.float32)
    data_nii = data.transpose(2, 1, 0)
    img = nib.Nifti1Image(data_nii, ref_affine)
    img.header.set_data_dtype(np.float32)
    if not path.endswith(".nii") and not path.endswith(".gz"):
        path = path + ".nii.gz"
    nib.save(img, path)


def save_warp_field(
    disp_mm: torch.Tensor,
    path: str,
    ref_affine: np.ndarray,
) -> None:
    """Save a (3, D, H, W) displacement field as a 4D NIfTI."""
    if isinstance(disp_mm, torch.Tensor):
        disp_np = disp_mm.detach().cpu().numpy()
    else:
        disp_np = np.asarray(disp_mm)
    # (3,D,H,W) -> (W,H,D,3) with component order (dW,dH,dD) = (dX,dY,dZ)
    disp_nii = np.transpose(disp_np[[2, 1, 0]], (3, 2, 1, 0)).astype(np.float32)
    img = nib.Nifti1Image(disp_nii, ref_affine)
    img.header.set_intent(1007)  # NIFTI_INTENT_VECTOR
    img.header.set_data_dtype(np.float32)
    if not path.endswith(".nii") and not path.endswith(".gz"):
        path = path + ".nii.gz"
    nib.save(img, path)

import torch.nn.functional as F

def resample_to_reference(
    moving_data: np.ndarray,
    moving_affine: np.ndarray,
    fixed_affine: np.ndarray,
    fixed_shape: Tuple[int, int, int],
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Resample `moving_data` onto the `fixed` image grid using affine transforms.
    """
    D, H, W = fixed_shape

    # Transform: fixed voxel -> world -> moving voxel
    fix2mov = np.linalg.solve(moving_affine, fixed_affine)  # (4,4)

    # Build grid of fixed voxel coordinates in (D, H, W) order
    d_idx = np.arange(D, dtype=np.float64)
    h_idx = np.arange(H, dtype=np.float64)
    w_idx = np.arange(W, dtype=np.float64)
    gd, gh, gw = np.meshgrid(d_idx, h_idx, w_idx, indexing="ij")

    # NIfTI affine operates on (X=W, Y=H, Z=D) coords
    ones = np.ones_like(gd)
    pts_fix = np.stack([gw, gh, gd, ones], axis=0).reshape(4, -1)  # (4, D*H*W)

    # Apply transform
    pts_mov_nii = fix2mov @ pts_fix   # (4, D*H*W)
    x_mov = pts_mov_nii[0].reshape(D, H, W)
    y_mov = pts_mov_nii[1].reshape(D, H, W)
    z_mov = pts_mov_nii[2].reshape(D, H, W)

    Dm, Hm, Wm = moving_data.shape
    x_norm = (x_mov / max(Wm - 1, 1)) * 2.0 - 1.0
    y_norm = (y_mov / max(Hm - 1, 1)) * 2.0 - 1.0
    z_norm = (z_mov / max(Dm - 1, 1)) * 2.0 - 1.0

    mov_t = torch.from_numpy(moving_data).float().unsqueeze(0).unsqueeze(0)
    grid = torch.stack([
        torch.from_numpy(x_norm.astype(np.float32)),
        torch.from_numpy(y_norm.astype(np.float32)),
        torch.from_numpy(z_norm.astype(np.float32)),
    ], dim=-1).unsqueeze(0).float()

    out = F.grid_sample(
        mov_t, grid,
        mode=interpolation,
        padding_mode="zeros",
        align_corners=True,
    )
    return out.squeeze().numpy()
