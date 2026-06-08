"""
register.py — Top-level coarse-to-fine registration with VFC + isoTV.

Workflow
--------
1. Build pyramids of the fixed/moving images (and label volumes if used).
2. At each level, build the per-level VFC kernel (radius in mm fixed,
   so its voxel size shrinks with the level resolution).
3. Compute the *fixed* edge map and VFC field once for the level.
4. Optimise control knots via L-BFGS (VFC alignment + isoTV).
5. Upsample knots to the next finer level (and optionally halve the
   control-point spacing — see ``cp_refinements``).
6. Return the final dense displacement and optionally a warped moving
   image / warped label volume.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import math
import torch
import torch.nn.functional as F

from .config import DEFAULT_PARAMS, merge
from .debug import dump_pass as _dump_pass
from .edges import image_edge_map, label_edge_map, hybrid_edge_map
from .metrics_vfc import vfc_field_from_edges
from .optimizer import optimise_level
from .pyramid import auto_nlvl, build_pyramid
from .regularizers import smooth_iso_tv  # noqa: F401  (re-export friendly)
from .vfc import build_vfc_kernel
from .warp import (
    forward_diff,  # noqa: F401
    knot_grid_size,
    knots_to_dense,
    warp_image,
    warp_image_from_knots,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RegistrationResult:
    knots_mm: torch.Tensor                  # finest-level knots
    disp_mm: torch.Tensor                   # (3, D, H, W) dense displacement
    warped_image: Optional[torch.Tensor] = None
    warped_labels: Optional[torch.Tensor] = None
    grid_spacing_final: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    losses_per_level: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _safe_quantile(flat: torch.Tensor, q: float) -> torch.Tensor:
    """torch.quantile rejects tensors larger than ~16M elements; fall back
    to kthvalue (exact, O(N)) for big volumes."""
    n = flat.numel()
    if n <= 16_000_000:
        return torch.quantile(flat, q)
    k = max(1, min(n, int(round(float(q) * (n - 1))) + 1))
    return torch.kthvalue(flat, k).values


def _percentile_clip(
    x: torch.Tensor, lo: float, hi: float
) -> torch.Tensor:
    flat = x.flatten()
    qlo = _safe_quantile(flat, lo)
    qhi = _safe_quantile(flat, hi)
    if float(qhi - qlo) < 1e-8:
        return x
    return (x.clamp(min=float(qlo), max=float(qhi)) - qlo) / (qhi - qlo)


def _halve_spacing(spacing: Sequence[float]) -> Tuple[float, float, float]:
    return tuple(max(1.0, round(float(s) / 2.0)) for s in spacing)


def _gaussian_blur_3d(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable isotropic Gaussian blur on a [C, D, H, W] volume, sigma in voxels."""
    if sigma is None or sigma <= 0.0:
        return volume
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1,
                     device=volume.device, dtype=volume.dtype)
    k1d = torch.exp(-0.5 * (x / sigma) ** 2)
    k1d = k1d / k1d.sum()
    C = volume.shape[0]
    out = volume.unsqueeze(0)  # [1, C, D, H, W]
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = k1d.numel()
        kk = k1d.view(*shape).expand(C, -1, -1, -1, -1)
        pad = [0, 0, 0]
        pad[axis] = radius
        out = F.conv3d(out, kk, padding=tuple(pad), groups=C)
    return out.squeeze(0)


def _build_edge_fn(
    edge_source: str,
    pix_res: torch.Tensor,
    use_kornia: bool,
    label_ids: Optional[Sequence[int]] = None,
    hybrid_weight: float = 0.5,
    labels_volume: Optional[torch.Tensor] = None,
):
    """
    Return a callable mapping a (D,H,W) intensity image (the *current*
    warped moving image) to a (D,H,W) edge map. For ``labels``/``hybrid``
    modes the moving label map is also warped externally and fed via a
    closure-captured reference.
    """
    if edge_source == "image":
        return lambda img: image_edge_map(img, pix_res, use_kornia=use_kornia)

    if edge_source == "labels":
        if labels_volume is None:
            raise ValueError("edge_source='labels' requires a label volume.")
        # The closure ignores `img` and uses the (already-warped) labels.
        return lambda _img: label_edge_map(labels_volume, pix_res, label_ids=label_ids)

    if edge_source == "hybrid":
        if labels_volume is None:
            raise ValueError("edge_source='hybrid' requires a label volume.")
        return lambda img: hybrid_edge_map(
            image_edge_map(img, pix_res, use_kornia=use_kornia),
            label_edge_map(labels_volume, pix_res, label_ids=label_ids),
            weight=hybrid_weight,
        )

    raise ValueError(f"Unknown edge_source: {edge_source!r}")


def _upsample_knots(
    knots_mm: torch.Tensor,
    new_size: Tuple[int, int, int],
) -> torch.Tensor:
    return F.interpolate(
        knots_mm.unsqueeze(0),
        size=new_size,
        mode="trilinear",
        align_corners=True,
    ).squeeze(0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def register(
    fixed_image: torch.Tensor,
    moving_image: torch.Tensor,
    pix_res: Sequence[float],
    fixed_labels: Optional[torch.Tensor] = None,
    moving_labels: Optional[torch.Tensor] = None,
    label_ids: Optional[Sequence[int]] = None,
    mask: Optional[torch.Tensor] = None,
    return_warped: bool = True,
    return_warped_labels: bool = False,
    **params,
) -> RegistrationResult:
    """
    Run VFC + isoTV registration.

    Parameters
    ----------
    fixed_image, moving_image : (D, H, W) float tensors.
    pix_res                   : (3,) voxel spacing in mm (D, H, W order).
    fixed_labels, moving_labels : optional (D, H, W) integer label volumes
        (required when ``edge_source`` is ``"labels"`` or ``"hybrid"``).
    label_ids : optional explicit list of label IDs to include.
    mask      : optional (D, H, W) weight mask for the similarity term.
    return_warped : whether to compute and return the warped moving image.
    return_warped_labels : whether to also nearest-neighbour warp the moving labels.

    Other keyword arguments override values in
    :data:`ptvreg_vfc.config.DEFAULT_PARAMS`.

    Returns
    -------
    :class:`RegistrationResult`.
    """
    p = merge(DEFAULT_PARAMS, params)
    device = _resolve_device(p["device"])
    dtype = torch.float32

    fix = fixed_image.to(device=device, dtype=dtype)
    mov = moving_image.to(device=device, dtype=dtype)
    pr = torch.as_tensor(pix_res, device=device, dtype=dtype)

    if p["edge_source"] == "image":
        fix = _percentile_clip(fix, p["percentile_lo"], p["percentile_hi"])
        mov = _percentile_clip(mov, p["percentile_lo"], p["percentile_hi"])

    grid_spacing = tuple(float(s) for s in p["grid_spacing"])
    nlvl = p["nlvl"] or auto_nlvl(fix.shape, grid_spacing, k_down=p["k_down"])
    if p["verbose"]:
        print(f"[ptvreg_vfc] nlvl={nlvl}  base grid_spacing={grid_spacing}  device={device}")

    fix_pyr = build_pyramid(fix, nlvl, pr, k_down=p["k_down"], mode="trilinear")
    mov_pyr = build_pyramid(mov, nlvl, pr, k_down=p["k_down"], mode="trilinear")

    need_labels = p["edge_source"] in ("labels", "hybrid")
    if need_labels:
        if fixed_labels is None or moving_labels is None:
            raise ValueError(
                "edge_source 'labels'/'hybrid' requires both fixed_labels "
                "and moving_labels."
            )
        fl = fixed_labels.to(device=device).float()
        ml = moving_labels.to(device=device).float()
        flbl_pyr = build_pyramid(fl, nlvl, pr, k_down=p["k_down"], mode="nearest")
        mlbl_pyr = build_pyramid(ml, nlvl, pr, k_down=p["k_down"], mode="nearest")
    else:
        flbl_pyr = mlbl_pyr = [None] * nlvl

    base_grid_spacing = grid_spacing
    current_grid_spacing = base_grid_spacing
    knots_mm: Optional[torch.Tensor] = None
    losses: list = []

    n_passes = nlvl + max(0, int(p["cp_refinements"]))
    coarse_iters = p.get("coarse_max_iters")
    base_iters = p["max_iters"]
    for pass_idx in range(n_passes):
        # Pyramid level for this pass (last `cp_refinements` passes stay at finest)
        lvl_idx = min(pass_idx, nlvl - 1)
        # Coarse passes can use a higher iter cap (cheap at low res, big global gains)
        is_finest_or_refine = (lvl_idx == nlvl - 1)
        iters_this_pass = base_iters if (is_finest_or_refine or coarse_iters is None) \
            else int(coarse_iters)
        p_pass = {**p, "max_iters": iters_this_pass}
        img_fix, pr_lvl = fix_pyr[lvl_idx]
        img_mov, _ = mov_pyr[lvl_idx]
        lbl_fix = flbl_pyr[lvl_idx][0] if flbl_pyr[lvl_idx] is not None else None
        lbl_mov = mlbl_pyr[lvl_idx][0] if mlbl_pyr[lvl_idx] is not None else None

        # Pre-finest passes use the *base* grid spacing; refinement passes halve it.
        if pass_idx < nlvl:
            pass_grid_spacing = base_grid_spacing
            current_grid_spacing = pass_grid_spacing
        else:
            current_grid_spacing = _halve_spacing(current_grid_spacing)
            pass_grid_spacing = current_grid_spacing

        kshape = knot_grid_size(img_fix.shape, pass_grid_spacing)

        if knots_mm is None:
            knots_mm = torch.zeros(
                (3, *kshape), device=device, dtype=dtype
            )
        else:
            knots_mm = _upsample_knots(knots_mm, kshape)

        # Kernel for this pyramid level (radius is mm-fixed, voxels shrink with level)
        kernel = build_vfc_kernel(
            radius_mm=p["vfc_radius_mm"],
            pix_res=pr_lvl,
            gamma=p["vfc_gamma"],
            device=device,
            dtype=dtype,
        )

        # Fixed VFC: computed *once* per pass
        edge_fn_fix = _build_edge_fn(
            p["edge_source"], pr_lvl, p["use_kornia"],
            label_ids=label_ids, hybrid_weight=p["hybrid_weight"],
            labels_volume=lbl_fix,
        )
        with torch.no_grad():
            fix_edges = edge_fn_fix(img_fix)
            fix_vfc = vfc_field_from_edges(fix_edges, kernel)

        # Per-iteration moving edge function. For label modes we have to
        # warp the labels using the *current* knots each closure call —
        # we encapsulate this by giving the optimiser a wrapper edge_fn
        # that, for label sources, ignores its image argument and instead
        # re-warps the moving label volume.
        if p["edge_source"] == "image":
            edge_fn = lambda warped_img: image_edge_map(
                warped_img, pr_lvl, use_kornia=p["use_kornia"]
            )
        else:
            # Closure capturing the *current* knots via `knots_ref` trick
            knots_ref = {"k": knots_mm}

            def edge_fn(warped_img, _ref=knots_ref, _lbl=lbl_mov, _pr=pr_lvl,
                        _ids=label_ids, _w=p["hybrid_weight"],
                        _src=p["edge_source"], _kornia=p["use_kornia"]):
                # We cannot read the live optimiser variable here, so we use
                # the warped intensity image's grid: warp labels with the
                # same displacement that warped the image.
                # The optimiser supplies warped_img already; reproducing
                # the warp on labels requires the same dense displacement.
                # We pass labels through a sibling wrapper below instead.
                raise RuntimeError("internal: label edge_fn requires wrapper")

        # For label modes we wire a custom optimiser closure via the
        # combined edge+warp helper below.
        if p["edge_source"] == "image":
            vfc_fn = lambda em: vfc_field_from_edges(em, kernel)
            knots_mm = optimise_level(
                knots_mm=knots_mm,
                fix_vfc=fix_vfc,
                moving_image=img_mov,
                edge_fn=edge_fn,
                vfc_fn=vfc_fn,
                pix_res=pr_lvl,
                grid_spacing=pass_grid_spacing,
                iso_tv=p["iso_tv"],
                csqrt=p["csqrt"],
                vfc_loss_scale=p["vfc_loss_scale"],
                jac_reg=p["jac_reg"],
                sign_invariant=p["sign_invariant"],
                mask=mask if (mask is None or mask.shape == img_fix.shape) else None,
                max_iters=iters_this_pass,
                history=p["lbfgs_history"],
                verbose=p["verbose"],
                level_tag=f"pass{pass_idx + 1}/{n_passes} lvl{lvl_idx} it{iters_this_pass}",
            )
        else:
            knots_mm = _optimise_level_labels(
                knots_mm=knots_mm,
                fix_vfc=fix_vfc,
                moving_image=img_mov,
                moving_labels=lbl_mov,
                fixed_labels=lbl_fix,
                kernel=kernel,
                pix_res=pr_lvl,
                grid_spacing=pass_grid_spacing,
                params=p_pass,
                label_ids=label_ids,
                mask=mask if (mask is None or mask.shape == img_fix.shape) else None,
                level_tag=f"pass{pass_idx + 1}/{n_passes} lvl{lvl_idx} it{iters_this_pass}",
            )

        losses.append(None)  # placeholder for richer telemetry later

        if p["verbose"]:
            print(f"[ptvreg_vfc] pass {pass_idx + 1}/{n_passes} done "
                  f"(lvl={lvl_idx}, grid={pass_grid_spacing}, knots={kshape})")

        # ---- Debug dump (post-pass) -------------------------------------
        if p["debug"]:
            with torch.no_grad():
                disp_lvl = knots_to_dense(knots_mm, img_fix.shape)
                warped_lvl = warp_image(img_mov, disp_lvl, pr_lvl,
                                        mode="bilinear", padding_mode="border")
                if p["edge_source"] == "image":
                    edge_mov_dbg = image_edge_map(warped_lvl, pr_lvl,
                                                  use_kornia=p["use_kornia"])
                else:
                    # Warp moving labels by nearest-neighbour for the dump
                    lbl_warp = warp_image(lbl_mov, disp_lvl, pr_lvl,
                                          mode="nearest", padding_mode="border")
                    e_lbl = label_edge_map(lbl_warp, pr_lvl, label_ids=label_ids)
                    if p["edge_source"] == "hybrid":
                        e_img = image_edge_map(warped_lvl, pr_lvl,
                                               use_kornia=p["use_kornia"])
                        edge_mov_dbg = hybrid_edge_map(e_img, e_lbl,
                                                       weight=p["hybrid_weight"])
                    else:
                        edge_mov_dbg = e_lbl
                vfc_mov_dbg = vfc_field_from_edges(edge_mov_dbg, kernel)

                tag = (f"pass{pass_idx + 1:02d}_lvl{lvl_idx}"
                       f"_grid{int(round(pass_grid_spacing[0]))}")
                _dump_pass(
                    out_dir=p["debug"],
                    pass_tag=tag,
                    fix_img=img_fix,
                    mov_img=img_mov,
                    warped_img=warped_lvl,
                    edge_fix=fix_edges,
                    edge_mov=edge_mov_dbg,
                    vfc_fix=fix_vfc,
                    vfc_mov=vfc_mov_dbg,
                    disp=disp_lvl,
                    pix_res=pr_lvl,
                    affine=None,
                )
                if p["verbose"]:
                    print(f"  [debug] wrote {p['debug']}/{tag}/")

    # Final dense displacement at original resolution
    disp_mm = knots_to_dense(knots_mm, fixed_image.shape)

    warped = None
    if return_warped:
        warped = warp_image(mov, disp_mm, pr, mode="bilinear", padding_mode="border")

    warped_lbl = None
    if return_warped_labels and moving_labels is not None:
        ml_full = moving_labels.to(device=device).float()
        warped_lbl = warp_image(ml_full, disp_mm, pr,
                                mode="nearest", padding_mode="border")
        warped_lbl = warped_lbl.round().to(moving_labels.dtype)

    return RegistrationResult(
        knots_mm=knots_mm.detach().cpu(),
        disp_mm=disp_mm.detach().cpu(),
        warped_image=warped.detach().cpu() if warped is not None else None,
        warped_labels=warped_lbl.detach().cpu() if warped_lbl is not None else None,
        grid_spacing_final=tuple(float(s) for s in current_grid_spacing),
        losses_per_level=losses,
    )


# ---------------------------------------------------------------------------
# Label-mode level optimiser (needs to warp the label map every closure call)
# ---------------------------------------------------------------------------

def _optimise_level_labels(
    knots_mm: torch.Tensor,
    fix_vfc: torch.Tensor,
    moving_image: torch.Tensor,
    moving_labels: torch.Tensor,
    fixed_labels: Optional[torch.Tensor],
    kernel: torch.Tensor,
    pix_res: torch.Tensor,
    grid_spacing: Tuple[float, float, float],
    params: dict,
    label_ids: Optional[Sequence[int]],
    mask: Optional[torch.Tensor],
    level_tag: str,
) -> torch.Tensor:
    """Specialised optimisation when edges come from (warped) labels.

    The key difference vs the image-only path: ``F.grid_sample`` with
    ``mode='nearest'`` is *not* differentiable in the label values, but
    we don't need it to be — the autograd path that matters runs through
    the gradient of the (one-hot) masks of the *intensity-warped*
    moving image at the same displacement. We instead warp the labels
    *without* a gradient (just to build the edge map) and rely on the
    image-warp branch for the gradient signal. For pure ``labels`` mode
    we approximate by warping the labels via bilinear sampling on the
    one-hot stack — which is fully differentiable — and taking the
    gradient of those soft masks.
    """
    from .edges import _central_grad, hybrid_edge_map, image_edge_map
    from .warp import knots_to_dense as _k2d, warp_image as _warp

    k = knots_mm.clone().detach().requires_grad_(True)
    img_shape = tuple(moving_image.shape)

    # Knot spacing in mm
    sp = torch.empty(3, device=pix_res.device, dtype=pix_res.dtype)
    for i, gs in enumerate(grid_spacing):
        sp[i] = float(gs) * pix_res[i]

    # Pick the label IDs we'll actually use.
    if label_ids is None:
        unique, counts = torch.unique(moving_labels, return_counts=True)
        pairs = sorted(zip(unique.tolist(), counts.tolist()),
                       key=lambda p: -p[1])
        ids_sorted = [int(u) for u, _c in pairs if int(u) != 0]
    else:
        ids_sorted = [int(i) for i in label_ids]
    if not ids_sorted:
        return k.detach()

    # Optional cap: keep only the N largest labels by voxel count. Keeps
    # GPU memory bounded when the segmentation has dozens of classes.
    cap = params.get("max_labels")
    if cap is not None and len(ids_sorted) > int(cap):
        ids = ids_sorted[: int(cap)]
        if params.get("verbose"):
            print(f"  [{level_tag}] using top {len(ids)} of "
                  f"{len(ids_sorted)} labels by frequency")
    else:
        ids = ids_sorted

    # Force boosted IDs into the kept set even if they fell outside the cap.
    # ``dice_boost_ids`` may be a list of int IDs (uniform factor) or a dict
    # mapping id -> per-label multiplier.
    raw_boost = params.get("dice_boost_ids") or []
    if isinstance(raw_boost, dict):
        boost_factor_map = {int(k_): float(v_) for k_, v_ in raw_boost.items()}
    else:
        boost_factor_map = {int(b): None for b in raw_boost}  # None=use default
    boost_ids = list(boost_factor_map.keys())
    for b in boost_ids:
        if b in ids_sorted and b not in ids:
            ids.append(b)

    # Precompute one-hot masks once per level (no autograd needed on these).
    onehot = torch.stack(
        [(moving_labels == lid).float() for lid in ids], dim=0
    ).contiguous()
    onehot.requires_grad_(False)

    # Optional Gaussian-smooth one-hot masks. Soft boundaries kill the
    # voxel-scale Dice gradient noise that makes warps wiggle at label
    # surfaces. Sigma is in voxels (per-level, so physical extent scales
    # with the pyramid).
    dice_sigma = float(params.get("dice_sigma") or 0.0)
    if dice_sigma > 0.0:
        onehot = _gaussian_blur_3d(onehot, dice_sigma)
        if params.get("verbose"):
            print(f"  [{level_tag}] one-hot blur sigma={dice_sigma:g} vox "
                  f"(physical {dice_sigma * float(pix_res.mean()):.2f} mm)")

    # ---- Dice term setup ------------------------------------------------
    dice_weight = float(params.get("dice_weight") or 0.0)
    dice_boost  = float(params.get("dice_boost_factor") or 1.0)
    dice_min_vol = int(params.get("dice_min_vol") or 0)

    onehot_fix = None
    dice_w_per_label: list = []
    if dice_weight > 0.0 and fixed_labels is not None:
        fix_masks = []
        kept_idx: list = []
        kept_ids: list = []
        for ci, lid in enumerate(ids):
            sf = (fixed_labels == lid).float()
            sm_vol = float(onehot[ci].sum())
            sf_vol = float(sf.sum())
            if sf_vol < dice_min_vol or sm_vol < dice_min_vol:
                continue
            fix_masks.append(sf)
            kept_idx.append(ci)
            kept_ids.append(int(lid))
        if fix_masks:
            onehot_fix = torch.stack(fix_masks, dim=0).contiguous()
            onehot_fix.requires_grad_(False)
            if dice_sigma > 0.0:
                onehot_fix = _gaussian_blur_3d(onehot_fix, dice_sigma)
            base = dice_weight / len(kept_ids)
            def _factor_for(lid):
                if lid not in boost_factor_map:
                    return 1.0
                f = boost_factor_map[lid]
                return float(f) if f is not None else dice_boost
            dice_w_per_label = [base * _factor_for(lid) for lid in kept_ids]
            if params.get("verbose"):
                boosted = [(lid, _factor_for(lid))
                           for lid in kept_ids if lid in boost_factor_map]
                print(f"  [{level_tag}] dice on {len(kept_ids)} labels "
                      f"(base={base:.4f}, boosted={boosted})")
        else:
            kept_idx = []

    opt = torch.optim.LBFGS(
        [k], lr=1.0, max_iter=params["max_iters"],
        history_size=params["lbfgs_history"],
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7, tolerance_change=1e-12,
    )

    state = {"it": 0}

    use_ckpt = bool(params.get("label_checkpoint", True))
    kept_set = set(kept_idx) if onehot_fix is not None else set()
    dice_idx_map = ({ci: j for j, ci in enumerate(kept_idx)}
                    if onehot_fix is not None else {})

    def _label_block(onehot_c, sf_or_zero, dice_w, disp_in):
        # Single-label contribution to (sum_d (grad_d w)^2, dice_term).
        # Wrapped so gradient checkpointing can free the warp+grad
        # intermediates and recompute them in backward.
        w = _warp(onehot_c, disp_in, pix_res,
                  mode="bilinear", padding_mode="zeros")
        g = _central_grad(w, pix_res)
        sq = g.pow(2).sum(dim=0)
        # Dice contribution (zero if this label is not in kept set).
        inter = (sf_or_zero * w).sum()
        denom = sf_or_zero.sum() + w.sum()
        dice = dice_w * (1.0 - 2.0 * inter / (denom + 1e-6))
        return sq, dice

    def closure():
        opt.zero_grad()
        disp = _k2d(k, img_shape)

        # Edge magnitude = sqrt( sum_l sum_d (grad_d soft_l)^2 ).
        # Per-label work is checkpointed: warp + grad intermediates are
        # freed after the forward pass and recomputed in backward, so
        # peak VRAM is ~O(1 label) instead of O(L labels) for large L.
        accum = torch.zeros(img_shape, device=k.device, dtype=k.dtype)
        dice_loss = torch.zeros((), device=k.device, dtype=k.dtype)
        _zero_mask = torch.zeros((), device=k.device, dtype=k.dtype)
        for c in range(onehot.shape[0]):
            if c in kept_set:
                j = dice_idx_map[c]
                sf_c = onehot_fix[j]
                dw = dice_w_per_label[j]
            else:
                sf_c = _zero_mask  # broadcasts; contributes 0 to dice
                dw = 0.0
            if use_ckpt:
                from torch.utils.checkpoint import checkpoint
                sq_c, dice_c = checkpoint(
                    _label_block, onehot[c], sf_c, dw, disp,
                    use_reentrant=False,
                )
            else:
                sq_c, dice_c = _label_block(onehot[c], sf_c, dw, disp)
            accum = accum + sq_c
            dice_loss = dice_loss + dice_c
        eps = 1e-8
        edge_lbl = (accum + eps).sqrt()

        if params["edge_source"] == "hybrid":
            warped_img = _warp(moving_image, disp, pix_res,
                               mode="bilinear", padding_mode="border")
            edge_img = image_edge_map(warped_img, pix_res,
                                      use_kornia=params["use_kornia"])
            edge = hybrid_edge_map(edge_img, edge_lbl,
                                   weight=params["hybrid_weight"])
        else:
            edge = edge_lbl

        v_mov = vfc_field_from_edges(edge, kernel)
        from .metrics_vfc import vfc_alignment_loss as _align
        loss_a = _align(fix_vfc, v_mov, mask=mask,
                        sign_invariant=params["sign_invariant"])
        loss_t = smooth_iso_tv(k, sp, csqrt=params["csqrt"])
        loss = params["vfc_loss_scale"] * loss_a + params["iso_tv"] * loss_t
        if onehot_fix is not None:
            loss = loss + dice_loss
        loss.backward()
        state["it"] += 1
        if params["verbose"] and (state["it"] == 1 or state["it"] % 5 == 0):
            gnorm = float(k.grad.norm()) if k.grad is not None else float("nan")
            gmax  = float(k.grad.abs().max()) if k.grad is not None else float("nan")
            kmax  = float(k.detach().abs().max())
            dice_str = (f"  dice={float(dice_loss.detach()):.4f}"
                        if onehot_fix is not None else "")
            print(f"  [{level_tag}] it={state['it']:3d}  "
                  f"L={float(loss.detach()):.5f}  align={float(loss_a.detach()):.4f}  "
                  f"tv={float(loss_t.detach()):.4f}{dice_str}  "
                  f"|g|={gnorm:.3e}  g_max={gmax:.3e}  |k|_max={kmax:.3f}mm")
        return loss

    opt.step(closure)
    return k.detach()
