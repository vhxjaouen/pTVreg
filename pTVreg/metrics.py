import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalCrossCorrelation(nn.Module):
    """
    Local Cross Correlation metric matching MATLAB's loc_cc_fftn behaviour.

    sigma is the Gaussian window width in PIXELS per axis (z, y, x order for 3-D).
    Pass a scalar for isotropic windows or a 3-element list for anisotropic ones.
    The kernel size follows the MATLAB formula: max(2*ceil(2*sigma)+1, 5).
    variance is stabilised with deps before the square-root (MATLAB default: 1e-5).
    The returned metric map is (1 - r), where r is the Pearson local correlation
    coefficient.  Optimum value is 0 (perfect positive correlation).
    """

    def __init__(self, sigma=2.1, dim=3, deps=1e-5):
        super().__init__()
        self.dim = dim
        self.deps = deps

        # Normalise sigma to a per-axis list; clip to minimum 0.8 px (MATLAB behaviour)
        if isinstance(sigma, (list, tuple, np.ndarray)):
            self.sigma = [max(0.8, float(s)) for s in list(sigma)[:dim]]
            while len(self.sigma) < dim:
                self.sigma.append(self.sigma[-1])
        else:
            self.sigma = [max(0.8, float(sigma))] * dim

        # MATLAB kernel size formula per axis: max(2*ceil(2*sigma)+1, 5)
        self.kernel_sizes = [
            max(2 * int(np.ceil(2.0 * s)) + 1, 5) for s in self.sigma
        ]

        # Build per-axis 1-D Gaussian kernels stored as non-trainable parameters
        self._kernels = nn.ParameterList()
        for s, ks in zip(self.sigma, self.kernel_sizes):
            coords = torch.arange(ks, dtype=torch.float32) - (ks - 1) / 2.0
            k1d = torch.exp(-coords ** 2 / (2.0 * s ** 2))
            k1d = k1d / k1d.sum()
            self._kernels.append(nn.Parameter(k1d, requires_grad=False))

    # ------------------------------------------------------------------
    def _conv_axis(self, x, axis):
        """Depthwise 1-D Gaussian convolution along the given spatial axis."""
        k1d = self._kernels[axis].to(x.device)
        ks = self.kernel_sizes[axis]
        pad = ks // 2
        C = x.shape[1]
        if self.dim == 3:
            if axis == 0:
                ker = k1d.view(1, 1, ks, 1, 1).expand(C, 1, ks, 1, 1).contiguous()
                return F.conv3d(x, ker, padding=(pad, 0, 0), groups=C)
            elif axis == 1:
                ker = k1d.view(1, 1, 1, ks, 1).expand(C, 1, 1, ks, 1).contiguous()
                return F.conv3d(x, ker, padding=(0, pad, 0), groups=C)
            else:
                ker = k1d.view(1, 1, 1, 1, ks).expand(C, 1, 1, 1, ks).contiguous()
                return F.conv3d(x, ker, padding=(0, 0, pad), groups=C)
        else:  # dim == 2
            if axis == 0:
                ker = k1d.view(1, 1, ks, 1).expand(C, 1, ks, 1).contiguous()
                return F.conv2d(x, ker, padding=(pad, 0), groups=C)
            else:
                ker = k1d.view(1, 1, 1, ks).expand(C, 1, 1, ks).contiguous()
                return F.conv2d(x, ker, padding=(0, pad), groups=C)

    def _smooth(self, x):
        out = x
        for ax in range(self.dim):
            out = self._conv_axis(out, ax)
        return out

    # ------------------------------------------------------------------
    def forward(self, fixed, moving, return_map=False):
        """fixed, moving: (N, C, [D,] H, W)"""
        I = fixed
        J = moving

        # Cache static terms for the fixed image
        if not hasattr(self, '_cached_fixed') or self._cached_fixed is not I:
            self._cached_fixed = I
            self._mu_I  = self._smooth(I)
            self._mu_I2 = self._smooth(I * I)

        mu_I  = self._mu_I
        mu_I2 = self._mu_I2

        mu_J   = self._smooth(J)
        mu_J2  = self._smooth(J * J)
        mu_IJ  = self._smooth(I * J)

        # Local std-devs with variance stabilisation before sqrt (matches MATLAB)
        # sgm = sqrt(E[x^2] - E[x]^2 + deps)
        sgm_I  = torch.sqrt(torch.clamp(mu_I2 - mu_I ** 2 + self.deps, min=1e-12))
        sgm_J  = torch.sqrt(torch.clamp(mu_J2 - mu_J ** 2 + self.deps, min=1e-12))
        cov_IJ = mu_IJ - mu_I * mu_J

        # Pearson local correlation coefficient r  (not r^2)
        r = cov_IJ / (sgm_I * sgm_J + 1e-10)

        # Loss = 1 - r,  range [0, 2],  optimum = 0
        cc_map = 1.0 - r

        return cc_map if return_map else torch.mean(cc_map)

def _grad_magnitude_3d(image_5d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Gradient magnitude of a (N, 1, D, H, W) image.
    Tries kornia.filters.SpatialGradient3d first (preferred); falls back to
    boundary-safe central differences.
    Returns (N, 1, D, H, W).
    """
    try:
        from kornia.filters import SpatialGradient3d
        # kornia output: (N, C, 3, D, H, W) — gradients in voxel units
        g = SpatialGradient3d()(image_5d)   # (N, 1, 3, D, H, W)
        gd, gh, gw = g[:, :, 0], g[:, :, 1], g[:, :, 2]
    except Exception:
        x = image_5d
        gd = F.pad(x, (0,0, 0,0, 1,1))[:, :, 2:, :, :] - F.pad(x, (0,0, 0,0, 1,1))[:, :, :-2, :, :]
        gh = F.pad(x, (0,0, 1,1, 0,0))[:, :, :, 2:, :] - F.pad(x, (0,0, 1,1, 0,0))[:, :, :, :-2, :]
        gw = F.pad(x, (1,1, 0,0, 0,0))[:, :, :, :, 2:] - F.pad(x, (1,1, 0,0, 0,0))[:, :, :, :, :-2]
    return torch.sqrt(gd**2 + gh**2 + gw**2 + eps)


class EdgeMSE(nn.Module):
    """MSE on gradient-magnitude maps. Uses kornia when available."""

    def __init__(self):
        super().__init__()

    def forward(self, img1, img2, return_map=False):
        mag1 = _grad_magnitude_3d(img1)
        mag2 = _grad_magnitude_3d(img2)
        emse = (mag1 - mag2) ** 2
        emse_norm = emse / (mag1.pow(2).mean() + 1e-8)
        return emse_norm if return_map else torch.mean(emse_norm)


# ---------------------------------------------------------------------------
# VFC helper functions — inlined from Li & Acton (2007).
# No ptvreg_vfc dependency; kornia is used for edge maps when available.
# ---------------------------------------------------------------------------

def _vfc_next_fft_size(n: int) -> int:
    """Round up to next power of two for FFT efficiency."""
    return 1 << max(1, (n - 1)).bit_length()


def _vfc_build_kernel(
    radius_mm: float,
    pix_res: torch.Tensor,
    gamma: float = 1.7,
    device=None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build a 3-D inward-pointing VFC kernel (Li & Acton 2007).

    Parameters
    ----------
    radius_mm : kernel radius in mm.
    pix_res   : (3,) voxel spacing in mm, order (D, H, W).
    gamma     : decay exponent.  Larger → more localised.
    device    : torch device for the returned tensor.
    eps       : singularity guard.

    Returns
    -------
    (3, Kd, Kh, Kw) float32 kernel, component order (vD, vH, vW),
    normalised so the peak absolute value per component is 1.
    """
    dtype = torch.float32
    pix_res = pix_res.to(device=device, dtype=dtype)

    max_pix = float(pix_res.max().item())
    eff_radius = float(radius_mm)
    if eff_radius < 1.01 * max_pix:
        warnings.warn(
            f"VFC radius {eff_radius:.3g} mm < voxel spacing {max_pix:.3g} mm; "
            f"clamping to {1.01*max_pix:.3g} mm so the kernel covers 6-neighbours."
        )
        eff_radius = 1.01 * max_pix

    r_vox = [max(1, int(math.ceil(eff_radius / float(pix_res[i].item())))) for i in range(3)]

    def _ax(n, sp):
        return torch.arange(-n, n + 1, device=device, dtype=dtype) * sp

    gd, gh, gw = torch.meshgrid(
        _ax(r_vox[0], float(pix_res[0])),
        _ax(r_vox[1], float(pix_res[1])),
        _ax(r_vox[2], float(pix_res[2])),
        indexing="ij",
    )  # each (Kd, Kh, Kw) in mm

    r = torch.sqrt(gd**2 + gh**2 + gw**2)
    mask = (r <= eff_radius).to(dtype)
    mag = mask / (r.pow(gamma + 1.0) + eps)
    mag = mag.masked_fill(r == 0, 0.0)        # no singularity at centre
    safe_r = r.clamp(min=eps)

    kernel = torch.stack([
        -gd / safe_r * mag,
        -gh / safe_r * mag,
        -gw / safe_r * mag,
    ], dim=0)  # (3, Kd, Kh, Kw)

    peak = kernel.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=eps)
    return kernel / peak


def _vfc_fft_field(edge_map: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Apply VFC to a (D, H, W) scalar edge map.
    kernel: (3, Kd, Kh, Kw) from _vfc_build_kernel.
    Returns (3, D, H, W) vector field.
    """
    D, H, W = edge_map.shape
    Kd, Kh, Kw = kernel.shape[1:]
    fft_s = (
        _vfc_next_fft_size(D + Kd - 1),
        _vfc_next_fft_size(H + Kh - 1),
        _vfc_next_fft_size(W + Kw - 1),
    )
    Fv = torch.fft.rfftn(edge_map, s=fft_s)
    sd, sh, sw = Kd // 2, Kh // 2, Kw // 2
    out = torch.empty(3, D, H, W, device=edge_map.device, dtype=edge_map.dtype)
    for c in range(3):
        Fk = torch.fft.rfftn(kernel[c], s=fft_s)
        conv = torch.fft.irfftn(Fv * Fk, s=fft_s)
        out[c] = conv[sd:sd + D, sh:sh + H, sw:sw + W]
    return out


def _vfc_image_edge_map(
    image: torch.Tensor,
    pix_res: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Gradient-magnitude edge map of a (D, H, W) intensity volume, in mm-aware units.
    Tries kornia.filters.SpatialGradient3d first (preferred); falls back to
    central differences.
    Returns (D, H, W).
    """
    pr = pix_res.to(image.device, dtype=image.dtype).view(3, 1, 1, 1)
    try:
        from kornia.filters import SpatialGradient3d
        # kornia: input (N,C,D,H,W), output (N,C,3,D,H,W) in voxel units
        g = SpatialGradient3d()(image[None, None])[0, 0]  # (3,D,H,W), voxel units
        grad = g / pr                                       # convert to mm units
    except Exception:
        grad = torch.stack([
            (F.pad(image, (0,0, 0,0, 1,1))[2:] - F.pad(image, (0,0, 0,0, 1,1))[:-2]) / (2.0 * pr[0]),
            (F.pad(image, (0,0, 1,1, 0,0))[:, 2:] - F.pad(image, (0,0, 1,1, 0,0))[:, :-2]) / (2.0 * pr[1]),
            (F.pad(image, (1,1, 0,0, 0,0))[:, :, 2:] - F.pad(image, (1,1, 0,0, 0,0))[:, :, :-2]) / (2.0 * pr[2]),
        ], dim=0)
    return (grad.pow(2).sum(dim=0) + eps).sqrt()

class VFCMetric(nn.Module):
    """
    Vector Field Convolution similarity metric (Li & Acton 2007).

    Fully self-contained — no ptvreg_vfc dependency.
    Kornia is used for edge-map extraction when available (preferred), with a
    central-difference fallback.

    The VFC kernel is rebuilt once per pyramid level from the physical voxel
    spacing.  At each level the fixed and moving VFC fields are pre-computed
    once via FFT convolution; during optimisation the moving VFC is *warped*
    (not recomputed), which is ~50× cheaper and introduces only second-order
    geometric error for the smooth fields produced by large-radius kernels.

    Loss = mean(1 − cos(vfc_fix, warp(vfc_mov, u)))
    If sign_invariant=True: loss = mean(1 − cos²(…))  ← good for label maps.
    """

    def __init__(self, radius_mm: float = 15.0, gamma: float = 1.7,
                 sign_invariant: bool = False):
        super().__init__()
        self.radius_mm = radius_mm
        self.gamma = gamma
        self.sign_invariant = sign_invariant
        self._kernel = None
        self._pix_res = None

    # ------------------------------------------------------------------
    def set_level(self, pix_res: list, device: str):
        """Rebuild the VFC kernel for the current pyramid level.
        pix_res: [pd, ph, pw] in mm (physical voxel spacing at this level).
        """
        pix_res_t = torch.tensor(pix_res, dtype=torch.float32, device=device)
        self._kernel = _vfc_build_kernel(
            self.radius_mm, pix_res_t, gamma=self.gamma, device=device
        )
        self._pix_res = pix_res_t

    # ------------------------------------------------------------------
    def _normalise_edge(self, edge: torch.Tensor) -> torch.Tensor:
        """Percentile-clip and normalise an edge map to [0, 1]."""
        flat = edge.flatten()
        n = flat.numel()
        k = max(1, min(n, int(round(0.999 * (n - 1))) + 1))
        e_max = torch.kthvalue(flat, k).values.clamp(min=1e-8)
        return (edge / e_max).clamp(0.0, 1.0)

    def edge_to_vfc(self, edge_map: torch.Tensor) -> torch.Tensor:
        """Convert a (D, H, W) edge-magnitude map to a (3, D, H, W) VFC field."""
        assert self._kernel is not None, "Call set_level() before edge_to_vfc()"
        return _vfc_fft_field(self._normalise_edge(edge_map), self._kernel)

    def image_to_vfc(self, image_3d: torch.Tensor) -> torch.Tensor:
        """Compute VFC field directly from a (D, H, W) intensity image."""
        assert self._pix_res is not None, "Call set_level() before image_to_vfc()"
        edge = _vfc_image_edge_map(image_3d, self._pix_res)
        return self.edge_to_vfc(edge)

    # ------------------------------------------------------------------
    def alignment_loss(
        self,
        fix_vfc: torch.Tensor,      # (3, D, H, W)
        warped_mov_vfc: torch.Tensor,  # (3, D, H, W)
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Returns (1, 1, D, H, W) per-voxel loss map = 1 - cos(fix, mov_warped).
        Shape matches the masking convention in PTVRegistration.compute_loss.
        """
        n_fix = fix_vfc / (fix_vfc.norm(dim=0, keepdim=True) + eps)
        n_mov = warped_mov_vfc / (warped_mov_vfc.norm(dim=0, keepdim=True) + eps)
        dot = (n_fix * n_mov).sum(dim=0)  # (D, H, W)
        if self.sign_invariant:
            dot = dot.pow(2)
        return (1.0 - dot).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)


class MultiChannelLCC(nn.Module):
    """
    Mean of per-channel Local Cross-Correlation over a C-channel image pair.

    Inputs are expected as (N, C, D, H, W).  Each channel is correlated
    independently; the results are averaged.  This makes it suitable for
    vector-valued images such as gradient fields, where each component
    carries independent directional information.
    """
    def __init__(self, sigma=2.1, dim=3):
        super().__init__()
        self.sigma = sigma
        self.dim = dim
        self._lcc = LocalCrossCorrelation(sigma=sigma, dim=dim)

    def forward(self, fixed, moving, return_map=False):
        # fixed, moving: (N, C, D, H, W)
        C = fixed.shape[1]
        channel_losses = []
        for c in range(C):
            f_c = fixed[:, c:c+1, ...]   # (N, 1, D, H, W)
            m_c = moving[:, c:c+1, ...]
            # Invalidate the internal fixed-image cache when channel changes
            if hasattr(self._lcc, '_cached_fixed'):
                del self._lcc._cached_fixed
            channel_losses.append(self._lcc(f_c, m_c, return_map=return_map))
        if return_map:
            return torch.mean(torch.stack(channel_losses, dim=0), dim=0)
        return torch.mean(torch.stack(channel_losses))