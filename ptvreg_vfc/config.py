"""
config.py — Defaults and JSON config helpers for ptvreg_vfc.
"""

from __future__ import annotations
import json, os
from copy import deepcopy
from typing import Any


DEFAULT_PARAMS: dict[str, Any] = {
    # ---- Edge / VFC ----
    "edge_source":      "image",       # "image" | "labels" | "hybrid"
    "hybrid_weight":    0.5,            # weight of label edges in hybrid mode
    "vfc_radius_mm":    10.0,
    "vfc_gamma":        1.7,
    "use_kornia":       False,
    # ---- Similarity ----
    "sign_invariant":   False,          # cosine if False, |cosine| if True
    "vfc_loss_scale":   100.0,
    "max_labels":       None,           # cap labels used in label/hybrid modes
    # ---- Per-label soft-Dice term (labels/hybrid modes only) ----
    # Total Dice weight shared equally across kept label pairs; 0 disables.
    "dice_weight":       0.0,
    # Gaussian blur (sigma in VOXELS) applied to one-hot masks before Dice.
    # 0 = off. Typical: 1.0-2.0. Smooths label boundaries so soft-Dice
    # gradients become smooth ramps -> eliminates voxel-scale warp wiggles
    # at the surface of large labels (e.g. body envelope).
    "dice_sigma":        0.0,
    # Label IDs whose share is multiplied by ``dice_boost_factor`` and which
    # are forced into the kept set regardless of ``max_labels``.
    "dice_boost_ids":    [],
    "dice_boost_factor": 3.0,
    # Minimum voxel count (per side) required to include a label in Dice.
    "dice_min_vol":      50,
    # Gradient-checkpoint each per-label warp in labels/hybrid mode to bound
    # peak VRAM at ~O(1 label) (trades a ~2x compute hit). Disable if you
    # have only a handful of labels and want maximum speed.
    "label_checkpoint":  True,
    # ---- Deformation ----
    "grid_spacing":     [8.0, 8.0, 8.0],
    # ---- Regularisation ----
    # Note: ``iso_tv`` is the literal coefficient in
    #   loss = vfc_loss_scale * align + iso_tv * tv
    # so when adjusting ``vfc_loss_scale``, scale ``iso_tv`` by the same
    # factor to preserve the alignment/regularisation balance.
    "iso_tv":           15.0,
    "csqrt":            0.01,
    "jac_reg":          0.0,
    # ---- Pyramid ----
    "nlvl":             None,
    "k_down":           0.7,
    # ---- Optimiser ----
    "max_iters":        80,
    "coarse_max_iters": None,    # per-pass L-BFGS cap for non-finest pyramid passes; None -> use max_iters
    "cp_refinements":   0,
    "lbfgs_history":    50,
    # ---- Preprocessing ----
    "percentile_hi":    0.99,
    "percentile_lo":    0.005,
    # ---- Execution ----
    "device":           "auto",         # "auto" | "cuda" | "cpu"
    "verbose":          True,
    # ---- Debug dumps ----
    "debug":            None,         # if set, dump per-pass artefacts here
}


def load_config(path: str) -> dict[str, Any]:
    with open(path) as fh:
        raw = json.load(fh)
    if "parameters" in raw and isinstance(raw["parameters"], dict):
        raw = raw["parameters"]
    return {k: v for k, v in raw.items() if k in DEFAULT_PARAMS}


def save_config(params: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(params, fh, indent=2)


def merge(*dicts: dict[str, Any]) -> dict[str, Any]:
    """Left-to-right merge skipping None-valued keys in later dicts."""
    out = deepcopy(dicts[0])
    for d in dicts[1:]:
        for k, v in d.items():
            if v is not None:
                out[k] = v
    return out
