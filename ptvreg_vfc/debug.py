"""
debug.py — Optional intermediate-state dumping for ptvreg_vfc.

Writes per-pass artefacts to disk for inspection:

* NIfTI:
    - ``edge_fix.nii.gz``, ``edge_mov.nii.gz``     — scalar edge maps
    - ``vfc_fix.nii.gz``, ``vfc_mov.nii.gz``       — 4D vector fields (D,H,W,3)
    - ``disp.nii.gz``                              — current dense displacement
    - ``warped.nii.gz``                            — current warped moving image

* RGB PNGs (3 orthogonal central planes), with the edge map as
  background and the VFC vector field mapped to RGB color channels
  (Z/D -> Red, Y/H -> Green, X/W -> Blue).

The matplotlib import is lazy — if it's missing we still write the
NIfTI files and just skip the PNGs (with a printed warning).
"""

from __future__ import annotations
import os
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _save_scalar_nii(arr: np.ndarray, path: str, affine: Optional[np.ndarray]) -> None:
    import nibabel as nib
    aff = affine if affine is not None else np.eye(4, dtype=np.float64)
    nii = arr.astype(np.float32).transpose(2, 1, 0)   # (D,H,W) -> (W,H,D)
    img = nib.Nifti1Image(nii, aff)
    img.header.set_data_dtype(np.float32)
    nib.save(img, path)


def _save_vector_nii(field: np.ndarray, path: str, affine: Optional[np.ndarray]) -> None:
    """field: (3,D,H,W) with component order (dD,dH,dW) -> NIfTI (W,H,D,3)."""
    import nibabel as nib
    aff = affine if affine is not None else np.eye(4, dtype=np.float64)
    # Reorder components to (dW,dH,dD) to align with NIfTI X,Y,Z convention.
    reordered = field[[2, 1, 0]]
    nii = np.transpose(reordered, (3, 2, 1, 0)).astype(np.float32)
    img = nib.Nifti1Image(nii, aff)
    img.header.set_intent(1007)   # NIFTI_INTENT_VECTOR
    img.header.set_data_dtype(np.float32)
    nib.save(img, path)


# ---------------------------------------------------------------------------
# RGB plotting
# ---------------------------------------------------------------------------

def vector_field_to_rgb(field: np.ndarray) -> np.ndarray:
    """
    Map vector field components to RGB.
    field: (3, H, W) slice where components are (dD, dH, dW).
    Maps Z (dD) -> Red, Y (dH) -> Green, X (dW) -> Blue.
    Magnitude is normalized per slice.
    """
    # Absolute value because colors can only be positive [0, 1]
    rgb = np.abs(field)
    
    # Normalize by max vector magnitude in the slice
    mag = np.linalg.norm(field, axis=0)
    max_mag = np.max(mag)
    if max_mag > 1e-9:
        rgb = rgb / max_mag
        
    # rgb currently is (3, H, W)
    # Transpose to (H, W, 3) for matplotlib
    return np.transpose(rgb, (1, 2, 0))


def _save_rgb_png(
    edge: np.ndarray,
    vfc: np.ndarray,
    path: str,
    suptitle: str,
) -> bool:
    """Render 3 orthogonal central-plane RGB panels. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  [debug] matplotlib unavailable ({exc}); skipping PNG {path}")
        return False

    D, H, W = edge.shape
    cz, cy, cx = D // 2, H // 2, W // 2

    # Axial (D-slice): plane is (H, W)
    bg_ax = edge[cz]
    vfc_ax = vfc[:, cz, :, :]
    
    # Coronal (H-slice): plane is (D, W)
    bg_co = edge[:, cy, :]
    vfc_co = vfc[:, :, cy, :]
    
    # Sagittal (W-slice): plane is (D, H)
    bg_sa = edge[:, :, cx]
    vfc_sa = vfc[:, :, :, cx]

    panels = []
    for bg, vfc_slice, tag in [
        (bg_ax, vfc_ax, f"axial z={cz}"),
        (bg_co, vfc_co, f"coronal y={cy}"),
        (bg_sa, vfc_sa, f"sagittal x={cx}"),
    ]:
        rgb = vector_field_to_rgb(vfc_slice)
        panels.append((bg, rgb, tag))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (bg, rgb, tag) in zip(axes, panels):
        # Overlay RGB on the edge background map
        # Convert bg to (H, W, 3) grayscale for blending if desired, or just show RGB.
        # The prompt says: "with the edge map as background and the in-plane components of the VFC vector field mapped to RGB color channels"
        # We can blend them: 50% bg, 50% RGB.
        bg_rgb = np.stack((bg, bg, bg), axis=-1)
        # Normalize bg to [0, 1] just in case
        if bg_rgb.max() > 1e-9:
            bg_rgb = bg_rgb / bg_rgb.max()
        
        blended = 0.4 * bg_rgb + 0.6 * rgb
        ax.imshow(blended, origin="lower", interpolation="nearest", aspect="equal")
        ax.set_title(tag, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def dump_pass(
    out_dir: str,
    pass_tag: str,
    fix_img: torch.Tensor,
    mov_img: torch.Tensor,
    warped_img: torch.Tensor,
    edge_fix: torch.Tensor,
    edge_mov: torch.Tensor,
    vfc_fix: torch.Tensor,
    vfc_mov: torch.Tensor,
    disp: torch.Tensor,
    pix_res: torch.Tensor,
    affine: Optional[np.ndarray] = None,
) -> None:
    """
    Dump intermediate state for a single pass.

    Files are written under ``out_dir/<pass_tag>/``.

    Parameters
    ----------
    out_dir       : root debug directory (created if missing).
    pass_tag      : subdirectory name, e.g. ``"pass03_lvl2"``.
    fix_img, mov_img, warped_img : (D,H,W) tensors at the current level.
    edge_fix, edge_mov           : (D,H,W) edge magnitude maps.
    vfc_fix, vfc_mov             : (3,D,H,W) VFC vector fields.
    disp                         : (3,D,H,W) dense displacement at this level.
    pix_res                      : (3,) voxel spacing in mm (unused for NIfTI
                                   affine when ``affine`` is supplied).
    affine                       : optional 4x4 voxel-to-world affine; if None
                                   identity is used.
    """
    pass_dir = os.path.join(out_dir, pass_tag)
    os.makedirs(pass_dir, exist_ok=True)

    # ---- NIfTI dumps -----------------------------------------------------
    fix_np    = _to_np(fix_img)
    mov_np    = _to_np(mov_img)
    warp_np   = _to_np(warped_img)
    edge_f_np = _to_np(edge_fix)
    edge_m_np = _to_np(edge_mov)
    vfc_f_np  = _to_np(vfc_fix)
    vfc_m_np  = _to_np(vfc_mov)
    disp_np   = _to_np(disp)

    _save_scalar_nii(fix_np,    os.path.join(pass_dir, "fixed.nii.gz"),    affine)
    _save_scalar_nii(mov_np,    os.path.join(pass_dir, "moving.nii.gz"),   affine)
    _save_scalar_nii(warp_np,   os.path.join(pass_dir, "warped.nii.gz"),   affine)
    _save_scalar_nii(edge_f_np, os.path.join(pass_dir, "edge_fix.nii.gz"), affine)
    _save_scalar_nii(edge_m_np, os.path.join(pass_dir, "edge_mov.nii.gz"), affine)
    _save_vector_nii(vfc_f_np,  os.path.join(pass_dir, "vfc_fix.nii.gz"),  affine)
    _save_vector_nii(vfc_m_np,  os.path.join(pass_dir, "vfc_mov.nii.gz"),  affine)
    _save_vector_nii(disp_np,   os.path.join(pass_dir, "disp.nii.gz"),     affine)

    # ---- RGB PNGs -----------------------------------------------------
    _save_rgb_png(
        edge_f_np, vfc_f_np,
        os.path.join(pass_dir, "rgb_fixed.png"),
        suptitle=f"{pass_tag} — fixed edge + VFC (RGB)",
    )
    _save_rgb_png(
        edge_m_np, vfc_m_np,
        os.path.join(pass_dir, "rgb_moving.png"),
        suptitle=f"{pass_tag} — moving edge + VFC (warped) (RGB)",
    )

    # ---- Tiny manifest ---------------------------------------------------
    with open(os.path.join(pass_dir, "shapes.txt"), "w") as fh:
        fh.write(f"image      : {tuple(fix_np.shape)}\n")
        fh.write(f"vfc fields : {tuple(vfc_f_np.shape)} (3,D,H,W)\n")
        fh.write(f"disp       : {tuple(disp_np.shape)} (3,D,H,W) in mm\n")
        fh.write(f"pix_res mm : {tuple(float(x) for x in pix_res.detach().cpu())}\n")
