"""
cli_register.py — Command-line wrapper around :func:`ptvreg_vfc.register.register`.

Example
-------
::

    ptvreg-vfc \\
        --fixed fixed.nii.gz --moving moving.nii.gz \\
        --warped warped.nii.gz --warp-field warp.nii.gz \\
        --edge-source labels \\
        --fixed-labels fix_seg.nii.gz --moving-labels mov_seg.nii.gz \\
        --vfc-radius-mm 12 --iso-tv 0.2 --max-iters 100
"""

from __future__ import annotations
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .config import DEFAULT_PARAMS, load_config, merge, save_config
from .io import load_label_nifti, load_nifti, save_nifti, save_warp_field
from .register import register
from .warp import warp_image_from_knots


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ptvreg-vfc",
        description="VFC + isotropic-TV deformable image registration",
    )
    p.add_argument("--fixed", help="Fixed image NIfTI (optional in labels mode).")
    p.add_argument("--moving", help="Moving image NIfTI (optional in labels mode).")
    p.add_argument("--warped", help="Output warped moving image (when --moving is used for registration).")
    p.add_argument("--warp-field", help="Output 4D displacement field NIfTI (opt-in; slow to write).")
    p.add_argument("--warped-labels", help="Output warped moving label volume.")
    p.add_argument("--moving-image",
                   help="Optional extra intensity image (not used for registration) to warp with the "
                        "computed knots and save to --warped-image. Useful in labels mode to also warp "
                        "the underlying MR/PET volume.")
    p.add_argument("--warped-image",
                   help="Output path for the warped --moving-image. Required when --moving-image is set.")
    p.add_argument("--fixed-labels", help="Fixed label NIfTI (needed for labels/hybrid).")
    p.add_argument("--moving-labels", help="Moving label NIfTI (needed for labels/hybrid).")
    p.add_argument("--label-ids", help="Comma-separated label IDs to use.")
    p.add_argument("--mask", help="Optional weight mask NIfTI.")
    p.add_argument("--config", help="JSON config to load defaults from.")
    p.add_argument("--save-config", help="Write the resolved config to this path.")
    p.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])

    # Knob mirrors of DEFAULT_PARAMS
    p.add_argument("--edge-source", default=None,
                   choices=["image", "labels", "hybrid"])
    p.add_argument("--hybrid-weight", type=float, default=None)
    p.add_argument("--vfc-radius-mm", type=float, default=None)
    p.add_argument("--vfc-gamma", type=float, default=None)
    p.add_argument("--use-kornia", action="store_true", default=None)
    p.add_argument("--sign-invariant", action="store_true", default=None)
    p.add_argument("--vfc-loss-scale", type=float, default=None)
    p.add_argument("--max-labels", type=int, default=None,
                   help="Cap on number of labels (largest by voxel count) used in labels/hybrid mode.")
    p.add_argument("--dice-weight", type=float, default=None,
                   help="Total soft-Dice weight, shared across all kept label pairs (0=off).")
    p.add_argument("--dice-sigma", type=float, default=None,
                   help="Gaussian blur (sigma in VOXELS) of one-hot masks before Dice. "
                        "0=off. Typical 1.0-2.0. Smooths label boundaries -> eliminates "
                        "voxel-scale warp wiggles at label surfaces.")
    p.add_argument("--dice-boost-ids", default=None,
                   help="Comma-separated label IDs whose Dice share is multiplied "
                        "by --dice-boost-factor. Per-label factors via 'id:factor' "
                        "syntax: '11:10,12:10,23:3,27:2'. Boosted IDs are kept regardless of --max-labels.")
    p.add_argument("--dice-boost-factor", type=float, default=None,
                   help="Multiplier applied to the per-label Dice weight for IDs in --dice-boost-ids.")
    p.add_argument("--dice-min-vol", type=int, default=None,
                   help="Minimum voxel count (per side) required for a label to enter the Dice term.")
    p.add_argument("--no-label-checkpoint", dest="label_checkpoint",
                   action="store_false", default=None,
                   help="Disable per-label gradient checkpointing (faster but uses ~Lx more VRAM).")
    p.add_argument("--grid-spacing", type=float, nargs=3, default=None,
                   metavar=("SD", "SH", "SW"))
    p.add_argument("--iso-tv", type=float, default=None)
    p.add_argument("--csqrt", type=float, default=None)
    p.add_argument("--jac-reg", type=float, default=None)
    p.add_argument("--nlvl", type=int, default=None)
    p.add_argument("--k-down", type=float, default=None)
    p.add_argument("--max-iters", type=int, default=None,
                   help="L-BFGS cap at the finest pyramid pass (and refinements).")
    p.add_argument("--coarse-max-iters", type=int, default=None,
                   help="L-BFGS cap for non-finest pyramid passes (coarse levels). "
                        "Defaults to --max-iters. Bigger value = better global alignment, cheap at low res.")
    p.add_argument("--cp-refinements", type=int, default=None)
    p.add_argument("--lbfgs-history", type=int, default=None)
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-iteration output.")
    p.add_argument("--debug", nargs="?", const="./debug_vfc", default=None,
                   help="Write per-pass NIfTI dumps and RGB PNGs here (default: ./debug_vfc if flag provided).")
    p.add_argument("--scale-factor", type=float, default=None,
                   help="Internally downsample inputs by this factor (0<sf<=1) "
                        "during optimisation; warp/labels output is produced at "
                        "original full resolution. E.g. 0.5 halves each axis.")
    return p


def _parse_boost_ids(spec: str | None):
    """Parse '11,12' (uniform factor) or '11:10,12:10,23:3' (per-label).

    Returns None if spec is empty, a list[int] for uniform, or a dict[int,float]
    for per-label factors.
    """
    if not spec:
        return None
    items = [s.strip() for s in spec.split(",") if s.strip()]
    if any(":" in it for it in items):
        out = {}
        for it in items:
            if ":" in it:
                k_, v_ = it.split(":", 1)
                out[int(k_)] = float(v_)
            else:
                out[int(it)] = None  # use default --dice-boost-factor
        return out
    return [int(it) for it in items]


def _ns_to_params(ns: argparse.Namespace) -> dict:
    cli = {
        "edge_source":    ns.edge_source,
        "hybrid_weight":  ns.hybrid_weight,
        "vfc_radius_mm":  ns.vfc_radius_mm,
        "vfc_gamma":      ns.vfc_gamma,
        "use_kornia":     ns.use_kornia,
        "sign_invariant": ns.sign_invariant,
        "vfc_loss_scale": ns.vfc_loss_scale,
        "max_labels":     ns.max_labels,
        "dice_weight":       ns.dice_weight,
        "dice_boost_ids":    _parse_boost_ids(ns.dice_boost_ids),
        "dice_boost_factor": ns.dice_boost_factor,
        "dice_min_vol":      ns.dice_min_vol,
        "label_checkpoint":  ns.label_checkpoint,
        "grid_spacing":   ns.grid_spacing,
        "iso_tv":         ns.iso_tv,
        "csqrt":          ns.csqrt,
        "jac_reg":        ns.jac_reg,
        "nlvl":           ns.nlvl,
        "k_down":         ns.k_down,
        "max_iters":      ns.max_iters,
        "cp_refinements": ns.cp_refinements,
        "lbfgs_history":  ns.lbfgs_history,
        "device":         ns.device,
        "verbose":        (not ns.quiet) if ns.quiet else None,
        "debug":          ns.debug,
    }
    return {k: v for k, v in cli.items() if v is not None}


def main(argv=None) -> int:
    ns = _build_parser().parse_args(argv)

    file_cfg = load_config(ns.config) if ns.config else {}
    cli_cfg = _ns_to_params(ns)
    params = merge(DEFAULT_PARAMS, file_cfg, cli_cfg)

    if ns.save_config:
        save_config(params, ns.save_config)

    fix_path = ns.fixed
    mov_path = ns.moving
    if params["edge_source"] == "labels":
        # In pure-labels mode the intensity image is not used by the
        # similarity term; fall back to the label files themselves so the
        # pyramid / warped-output plumbing still has something to work on.
        fix_path = fix_path or ns.fixed_labels
        mov_path = mov_path or ns.moving_labels
    if not fix_path or not mov_path:
        raise SystemExit("--fixed and --moving are required "
                         "(or use --edge-source labels with label files).")

    fix, fix_aff, fix_vox = load_nifti(fix_path)
    mov, _,      _       = load_nifti(mov_path)
    if fix.shape != mov.shape:
        print(f"warning: fixed {fix.shape} and moving {mov.shape} shapes differ;"
              " moving will be sampled into fixed grid via warp only.",
              file=sys.stderr)

    fix_t = torch.from_numpy(np.ascontiguousarray(fix))
    mov_t = torch.from_numpy(np.ascontiguousarray(mov))

    fixed_lbl_t = moving_lbl_t = None
    if params["edge_source"] in ("labels", "hybrid"):
        if not ns.fixed_labels or not ns.moving_labels:
            raise SystemExit(
                "--fixed-labels and --moving-labels are required for "
                f"edge_source={params['edge_source']!r}"
            )
        fl, _, _ = load_label_nifti(ns.fixed_labels)
        ml, _, _ = load_label_nifti(ns.moving_labels)
        fixed_lbl_t = torch.from_numpy(np.ascontiguousarray(fl))
        moving_lbl_t = torch.from_numpy(np.ascontiguousarray(ml))

    label_ids = None
    if ns.label_ids:
        label_ids = [int(x) for x in ns.label_ids.split(",") if x.strip()]

    mask_t = None
    if ns.mask:
        m, _, _ = load_nifti(ns.mask)
        mask_t = torch.from_numpy(np.ascontiguousarray(m))

    # Optional internal downsample for the optimisation pass.
    sf = ns.scale_factor
    if sf is not None and (sf <= 0.0 or sf > 1.0):
        raise SystemExit("--scale-factor must be in (0, 1].")
    full_shape = tuple(fix_t.shape)
    full_vox = fix_vox.astype(np.float32)
    fix_t_opt, mov_t_opt = fix_t, mov_t
    fixed_lbl_opt, moving_lbl_opt, mask_opt = fixed_lbl_t, moving_lbl_t, mask_t
    pix_res_opt = full_vox.tolist()

    if sf is not None and sf < 1.0:
        ds_shape = tuple(max(2, int(round(s * sf))) for s in full_shape)
        eff_sf = tuple(d / s for d, s in zip(ds_shape, full_shape))
        pix_res_opt = [float(full_vox[i]) / eff_sf[i] for i in range(3)]

        def _ds_img(t):
            x = t.float().unsqueeze(0).unsqueeze(0)
            return F.interpolate(x, size=ds_shape, mode="trilinear",
                                 align_corners=True).squeeze(0).squeeze(0)

        def _ds_lbl(t):
            x = t.float().unsqueeze(0).unsqueeze(0)
            return F.interpolate(x, size=ds_shape, mode="nearest"
                                 ).squeeze(0).squeeze(0).to(t.dtype)

        fix_t_opt = _ds_img(fix_t)
        mov_t_opt = _ds_img(mov_t)
        if fixed_lbl_t is not None:
            fixed_lbl_opt = _ds_lbl(fixed_lbl_t)
        if moving_lbl_t is not None:
            moving_lbl_opt = _ds_lbl(moving_lbl_t)
        if mask_t is not None:
            mask_opt = _ds_img(mask_t)

        if not ns.quiet:
            print(f"[scale-factor={sf}] optimising at {ds_shape} "
                  f"(voxel {tuple(round(v,3) for v in pix_res_opt)} mm) "
                  f"from full {full_shape} (voxel {tuple(round(float(v),3) for v in full_vox)} mm)")

    downsampled = sf is not None and sf < 1.0
    result = register(
        fixed_image=fix_t_opt,
        moving_image=mov_t_opt,
        pix_res=pix_res_opt,
        fixed_labels=fixed_lbl_opt,
        moving_labels=moving_lbl_opt,
        label_ids=label_ids,
        mask=mask_opt,
        # When downsampling, do the final warps ourselves at full-res.
        return_warped=bool(ns.warped) and not downsampled,
        return_warped_labels=(bool(ns.warped_labels)
                              and moving_lbl_t is not None
                              and not downsampled),
        **params,
    )

    # If we downsampled, produce full-resolution outputs from the knots.
    full_pix_res = torch.as_tensor(full_vox, dtype=result.knots_mm.dtype,
                                   device=result.knots_mm.device)
    if downsampled:
        if ns.warped:
            mov_full = mov_t.to(result.knots_mm.device,
                                dtype=result.knots_mm.dtype)
            with torch.no_grad():
                warped_full = warp_image_from_knots(
                    mov_full, result.knots_mm, full_pix_res,
                    mode="bilinear", padding_mode="border",
                )
            result.warped_image = warped_full.cpu()
        if ns.warped_labels and moving_lbl_t is not None:
            ml_full = moving_lbl_t.to(result.knots_mm.device,
                                      dtype=result.knots_mm.dtype)
            with torch.no_grad():
                wlbl_full = warp_image_from_knots(
                    ml_full, result.knots_mm, full_pix_res,
                    mode="nearest", padding_mode="border",
                )
            result.warped_labels = wlbl_full.cpu().to(moving_lbl_t.dtype)
        if ns.warp_field:
            from .warp import knots_to_dense
            with torch.no_grad():
                result.disp_mm = knots_to_dense(
                    result.knots_mm, full_shape
                ).cpu()

    if ns.warped and result.warped_image is not None:
        save_nifti(result.warped_image, ns.warped, fix_aff)
        print(f"wrote warped image -> {ns.warped}")
    if ns.warp_field:
        save_warp_field(result.disp_mm, ns.warp_field, fix_aff)
        print(f"wrote warp field   -> {ns.warp_field}")
    if ns.warped_labels and result.warped_labels is not None:
        save_nifti(result.warped_labels.float(), ns.warped_labels, fix_aff)
        print(f"wrote warped labels-> {ns.warped_labels}")

    if ns.moving_image:
        if not ns.warped_image:
            raise SystemExit("--moving-image requires --warped-image to write the result.")
        mi_np, mi_aff, _ = load_nifti(ns.moving_image)
        mi_t = torch.from_numpy(np.ascontiguousarray(mi_np)).to(
            result.knots_mm.device, dtype=result.knots_mm.dtype)
        with torch.no_grad():
            mi_warped = warp_image_from_knots(
                mi_t, result.knots_mm, full_pix_res,
                mode="bilinear", padding_mode="border",
            )
        save_nifti(mi_warped.cpu(), ns.warped_image, fix_aff)
        print(f"wrote warped image -> {ns.warped_image}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
