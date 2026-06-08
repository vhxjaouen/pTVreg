"""
cli_warp.py — Apply a saved DVF (from ptvreg-vfc) to a moving image.

Example
-------
::

    ptvwarp -i moving.nii.gz -w warp.nii.gz -r reference.nii.gz \\
            -o warped.nii.gz -int bsp3
"""

from __future__ import annotations
import argparse
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError("nibabel is required (pip install nibabel)") from exc


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _load_image(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (data_DHW, affine, voxel_mm_DHW)."""
    img = nib.load(path)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    data = data.transpose(2, 1, 0)
    affine = np.array(img.affine, dtype=np.float64)
    zooms = np.array(img.header.get_zooms()[:3], dtype=np.float32)[::-1].copy()
    return data, affine, zooms


def _load_warp(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a DVF saved by ``save_warp_field``.

    Returns
    -------
    disp_mm_DHW3 : (3, D, H, W) float32 displacement in mm,
                   components ordered (dD, dH, dW).
    affine       : (4, 4) reference affine the DVF was saved with.
    """
    img = nib.load(path)
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise SystemExit(f"warp NIfTI must be (W,H,D,3); got shape {arr.shape}")
    # File layout (set by save_warp_field):
    #   (W, H, D, 3)  with components (dX, dY, dZ).
    # Convert back to (3, D, H, W) with (dD, dH, dW).
    disp = np.transpose(arr, (3, 2, 1, 0))[[2, 1, 0]].astype(np.float32)
    return np.ascontiguousarray(disp), np.array(img.affine, dtype=np.float64)


def _save_image(data_DHW: np.ndarray, path: str, ref_affine: np.ndarray) -> None:
    out = np.asarray(data_DHW).astype(np.float32).transpose(2, 1, 0)
    img = nib.Nifti1Image(out, ref_affine)
    img.header.set_data_dtype(np.float32)
    if not path.endswith(".nii") and not path.endswith(".gz"):
        path = path + ".nii.gz"
    nib.save(img, path)


# ---------------------------------------------------------------------------
# Resample DVF onto an arbitrary reference grid (assumes same physical space).
# ---------------------------------------------------------------------------

def _resample_disp(disp_DHW3: np.ndarray,
                   ref_shape: Tuple[int, int, int]) -> np.ndarray:
    """Trilinear resample a (3,D,H,W) displacement to ``ref_shape``.

    The displacement components themselves stay in mm — only the spatial
    sampling changes. This assumes the DVF and reference grid cover the
    same physical volume (same world extent), which is the common case
    when the reference matches the registration's fixed image.
    """
    if tuple(disp_DHW3.shape[1:]) == tuple(ref_shape):
        return disp_DHW3
    t = torch.from_numpy(disp_DHW3).unsqueeze(0)   # (1,3,D,H,W)
    out = F.interpolate(t, size=ref_shape, mode="trilinear",
                        align_corners=True).squeeze(0).numpy()
    return np.ascontiguousarray(out.astype(np.float32))


# ---------------------------------------------------------------------------
# Interpolators
# ---------------------------------------------------------------------------

def _warp_grid_sample(image_DHW: np.ndarray,
                      disp_DHW3: np.ndarray,
                      pix_res_DHW: np.ndarray,
                      mode: str,
                      padding_mode: str = "border",
                      device: str = "cpu") -> np.ndarray:
    """Use torch.grid_sample for ``nearest`` / ``bilinear`` (i.e. trilinear)."""
    dev = torch.device(device)
    img = torch.from_numpy(image_DHW).to(dev).float()
    disp = torch.from_numpy(disp_DHW3).to(dev).float()
    pr = torch.as_tensor(pix_res_DHW, dtype=torch.float32, device=dev)

    D, H, W = img.shape
    disp_pix = disp / pr.view(3, 1, 1, 1)
    zz = torch.linspace(-1.0, 1.0, D, device=dev)
    yy = torch.linspace(-1.0, 1.0, H, device=dev)
    xx = torch.linspace(-1.0, 1.0, W, device=dev)
    gz, gy, gx = torch.meshgrid(zz, yy, xx, indexing="ij")
    dz = gz + disp_pix[0] * (2.0 / max(D - 1, 1))
    dy = gy + disp_pix[1] * (2.0 / max(H - 1, 1))
    dx = gx + disp_pix[2] * (2.0 / max(W - 1, 1))
    grid = torch.stack([dx, dy, dz], dim=-1).unsqueeze(0)
    out = F.grid_sample(img.view(1, 1, D, H, W), grid,
                        mode=mode, padding_mode=padding_mode,
                        align_corners=True)
    return out.view(D, H, W).cpu().numpy()


def _warp_bspline3(image_DHW: np.ndarray,
                   disp_DHW3: np.ndarray,
                   pix_res_DHW: np.ndarray) -> np.ndarray:
    """Cubic-spline interpolation (order=3) via scipy.ndimage.map_coordinates.

    Implements the same convention as ``warp_image``:
    ``warped[x] = image[x + disp(x)]``, with ``disp`` in mm.
    """
    try:
        from scipy.ndimage import map_coordinates
    except ImportError as exc:
        raise SystemExit("bsp3 interpolation requires scipy "
                         "(pip install scipy).") from exc

    D, H, W = image_DHW.shape
    # disp_pix per axis in (D,H,W) voxel offsets
    disp_pix = disp_DHW3 / np.asarray(pix_res_DHW,
                                      dtype=np.float32).reshape(3, 1, 1, 1)
    iz = np.arange(D, dtype=np.float32)[:, None, None] + disp_pix[0]
    iy = np.arange(H, dtype=np.float32)[None, :, None] + disp_pix[1]
    ix = np.arange(W, dtype=np.float32)[None, None, :] + disp_pix[2]
    coords = np.stack([iz, iy, ix], axis=0)
    return map_coordinates(image_DHW.astype(np.float32), coords,
                           order=3, mode="nearest").astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ptvwarp",
        description="Apply a saved DVF to an image with NN / linear / bspline3 "
                    "interpolation. Resamples onto the reference grid.",
    )
    p.add_argument("-i", "--input", required=True,
                   help="Moving image NIfTI to be warped.")
    p.add_argument("-w", "--warp", required=True,
                   help="DVF NIfTI written by ptvreg-vfc.")
    p.add_argument("-r", "--reference",
                   help="Reference image whose grid (shape + affine) the output "
                        "is resampled onto. Defaults to the input grid.")
    p.add_argument("-o", "--output", required=True,
                   help="Output warped image NIfTI.")
    p.add_argument("-int", "--interp", default="linear",
                   choices=["nn", "nearest", "linear", "trilinear",
                            "bsp3", "bspline3", "cubic"],
                   help="Interpolation mode (default: linear).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Device for trilinear / NN warps (default: cpu).")
    p.add_argument("--padding-mode", default="border",
                   choices=["border", "zeros", "reflection"],
                   help="Out-of-bounds sampling behaviour (default: border).")
    return p


_INTERP_ALIAS = {
    "nn": "nearest", "nearest": "nearest",
    "linear": "bilinear", "trilinear": "bilinear",
    "bsp3": "bsp3", "bspline3": "bsp3", "cubic": "bsp3",
}


def main(argv=None) -> int:
    ns = _build_parser().parse_args(argv)

    mov, mov_aff, mov_vox = _load_image(ns.input)
    disp, disp_aff = _load_warp(ns.warp)

    if ns.reference:
        ref_data, ref_aff, ref_vox = _load_image(ns.reference)
        ref_shape = ref_data.shape
        out_aff = ref_aff
        pix_res = ref_vox
        # Resample the saved DVF onto the reference grid.
        disp = _resample_disp(disp, ref_shape)
        # If the moving image is on a different grid than the reference,
        # resample it onto the reference first (linear) so we can apply
        # the DVF in the reference frame.
        if mov.shape != ref_shape:
            t = torch.from_numpy(mov).unsqueeze(0).unsqueeze(0).float()
            mov = F.interpolate(t, size=ref_shape, mode="trilinear",
                                align_corners=True
                                ).squeeze(0).squeeze(0).numpy()
            mov = np.ascontiguousarray(mov.astype(np.float32))
    else:
        ref_shape = mov.shape
        out_aff = mov_aff
        pix_res = mov_vox
        # Resample DVF onto moving grid if needed.
        disp = _resample_disp(disp, ref_shape)

    mode = _INTERP_ALIAS[ns.interp]
    if mode == "bsp3":
        out = _warp_bspline3(mov, disp, pix_res)
    else:
        out = _warp_grid_sample(mov, disp, pix_res, mode=mode,
                                padding_mode=ns.padding_mode,
                                device=ns.device)

    _save_image(out, ns.output, out_aff)
    print(f"wrote -> {ns.output}  shape={out.shape}  interp={ns.interp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
