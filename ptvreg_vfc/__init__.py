"""
ptvreg_vfc — Vector Field Convolution + isotropic Total Variation
registration in PyTorch.

Public surface
--------------
* :func:`register`           — end-to-end coarse-to-fine registration
* :class:`RegistrationResult` — output container
* :func:`build_vfc_kernel`, :func:`vfc_field`, :func:`normalize_vfc`
* :func:`image_edge_map`, :func:`label_edge_map`, :func:`hybrid_edge_map`
* :func:`vfc_alignment_loss`
* :func:`smooth_iso_tv`
* :func:`knots_to_dense`, :func:`warp_image`, :func:`warp_image_from_knots`
* :data:`DEFAULT_PARAMS`
"""

from .config import DEFAULT_PARAMS, load_config, save_config
from .edges import hybrid_edge_map, image_edge_map, label_edge_map
from .metrics_vfc import vfc_alignment_loss, vfc_field_from_edges
from .register import RegistrationResult, register
from .regularizers import smooth_iso_tv
from .vfc import build_vfc_kernel, normalize_vfc, vfc_field
from .warp import knot_grid_size, knots_to_dense, warp_image, warp_image_from_knots

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_PARAMS",
    "RegistrationResult",
    "build_vfc_kernel",
    "hybrid_edge_map",
    "image_edge_map",
    "knot_grid_size",
    "knots_to_dense",
    "label_edge_map",
    "load_config",
    "normalize_vfc",
    "register",
    "save_config",
    "smooth_iso_tv",
    "vfc_alignment_loss",
    "vfc_field",
    "vfc_field_from_edges",
    "warp_image",
    "warp_image_from_knots",
]
