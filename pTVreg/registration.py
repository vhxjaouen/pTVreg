import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .deformation import knots_to_dense_flow, warp_image
from .utils import to_torch, to_numpy
from .metrics import LocalCrossCorrelation, EdgeMSE, VFCMetric, MultiChannelLCC

class PTVRegistration(nn.Module):
    def __init__(self, fixed_image, moving_image, fixed_mask=None, border_mask=5,
                 grid_spacing=4, pix_resolution=1.0, device='cpu',
                 lambda_reg=0.11, n_levels=None, metric='lcc', metric_param=2.1, k_down=0.7,
                 fixed_edgemap=None, moving_edgemap=None,
                 vfc_radius=15.0, vfc_gamma=1.7, vfc_sign_invariant=False):
        """
        Args:
            fixed_image: (D, H, W) or (N, D, H, W) or None (for groupwise)
            moving_image: (D, H, W) or (N, D, H, W)
            grid_spacing: int, spacing of control points at finest level
            lambda_reg: regularization weight (default 0.11 for COPD)
            n_levels: number of pyramid levels, or None to auto-compute from image size
                      (matches MATLAB's automatic level calculation)
            metric: 'ssd', 'lcc', 'vfc', 'emse', 'nuclear', 'mclcc'
            metric_param: LCC sigma in physical units (mm); scaled to pixels per level
            fixed_edgemap: optional (D, H, W) pre-computed edge magnitude for fixed image (VFC)
            moving_edgemap: optional (D, H, W) pre-computed edge magnitude for moving image (VFC)
            vfc_radius: VFC kernel radius in mm (default 15.0)
            vfc_gamma: VFC kernel decay exponent (default 1.7)
            vfc_sign_invariant: use cos^2 alignment loss (good for label-derived edge maps)
        """
        super().__init__()
        self.device = device
        self.grid_spacing = grid_spacing
        self.k_down = k_down
        self.border_mask = border_mask
        self.metric = metric

        # Store metric_param in physical units for per-level pixel-space conversion
        if isinstance(metric_param, (list, tuple, np.ndarray)):
            self.metric_param_mm = [float(v) for v in list(metric_param)[:3]]
            while len(self.metric_param_mm) < 3:
                self.metric_param_mm.append(self.metric_param_mm[-1])
        else:
            self.metric_param_mm = [float(metric_param)] * 3

        if isinstance(pix_resolution, (list, tuple, np.ndarray)):
            if len(pix_resolution) == 1:
                self.pix_resolution = [float(pix_resolution[0])] * 3
            elif len(pix_resolution) >= 3:
                self.pix_resolution = [float(pix_resolution[0]), float(pix_resolution[1]), float(pix_resolution[2])]
            else:
                raise ValueError("pix_resolution must be a scalar or have 3 elements")
        else:
            self.pix_resolution = [float(pix_resolution)] * 3

        # Load moving image early so we can derive n_levels from the image shape
        self.moving = to_torch(moving_image, device=device)
        if self.moving.ndim == 3:
            self.moving = self.moving[None, None, ...]  # (1, 1, D, H, W)
        elif self.moving.ndim == 4:
            if self.moving.shape[-1] <= 9 and self.moving.shape[0] > 9:
                self.moving = self.moving.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, D, H, W)
            else:
                self.moving = self.moving[:, None, ...]  # (N, 1, D, H, W)

        # Auto-compute n_levels from image size when not provided (matches MATLAB formula)
        imsz = self.moving.shape[2:]
        if n_levels is None:
            n_levels = self._auto_nlvl(imsz, grid_spacing, k_down)
            print(f"Auto-computed n_levels={n_levels} for image size {tuple(imsz)}")
        self.n_levels = n_levels

        # Process lambda_reg; pad/truncate list to match n_levels
        if isinstance(lambda_reg, (list, tuple)):
            lr_list = [float(v) for v in lambda_reg]
            if len(lr_list) < self.n_levels:
                lr_list += [lr_list[-1]] * (self.n_levels - len(lr_list))
            self.lambda_reg = lr_list[:self.n_levels]
        else:
            self.lambda_reg = [float(lambda_reg)] * self.n_levels

        # Handle fixed mask
        if fixed_mask is not None:
            mask_t = to_torch(fixed_mask, device=device, dtype=torch.float32)
            if mask_t.ndim == 3:
                mask_t = mask_t.unsqueeze(0).unsqueeze(0)
            elif mask_t.ndim == 4:
                mask_t = mask_t.unsqueeze(0)
            self.fixed_mask = mask_t
        else:
            self.fixed_mask = None

        if fixed_image is not None:
            self.fixed = to_torch(fixed_image, device=device)
            if self.fixed.ndim == 3:
                self.fixed = self.fixed[None, None, ...]
            elif self.fixed.ndim == 4:
                if self.fixed.shape[-1] <= 9 and self.fixed.shape[0] > 9:
                    self.fixed = self.fixed.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, D, H, W)
                else:
                    self.fixed = self.fixed[:, None, ...]
            self.fixed_pyramid = self._create_pyramid(self.fixed, n_levels)
        else:
            self.fixed = None
            self.fixed_pyramid = None

        # Non-LCC metrics are fixed at construction; LCC/VFC are rebuilt per level in optimize()
        self.metric_fn = None
        if self.metric == 'mclcc':
            self.metric_fn = MultiChannelLCC(sigma=self.metric_param_mm[0], dim=3).to(device)
        elif self.metric == 'emse':
            self.metric_fn = EdgeMSE().to(device)
        elif self.metric == 'vfc':
            self.vfc_metric = VFCMetric(
                radius_mm=vfc_radius,
                gamma=vfc_gamma,
                sign_invariant=vfc_sign_invariant,
            ).to(device)
            self._vfc_radius = vfc_radius
            self._vfc_gamma = vfc_gamma
            self._vfc_sign_invariant = vfc_sign_invariant

        self.moving_pyramid = self._create_pyramid(self.moving, n_levels)

        # Build edgemap pyramids for VFC if user supplied pre-computed maps
        self.fixed_edgemap_pyramid = None
        self.moving_edgemap_pyramid = None
        if fixed_edgemap is not None:
            fe_t = to_torch(fixed_edgemap, device=device)
            if fe_t.ndim == 3:
                fe_t = fe_t[None, None, ...]
            self.fixed_edgemap_pyramid = self._create_pyramid(fe_t, self.n_levels)
        if moving_edgemap is not None:
            me_t = to_torch(moving_edgemap, device=device)
            if me_t.ndim == 3:
                me_t = me_t[None, None, ...]
            self.moving_edgemap_pyramid = self._create_pyramid(me_t, self.n_levels)

        # Per-level VFC caches (set in optimize(), used in compute_loss())
        self._cur_fix_vfc = None
        self._cur_mov_vfc = None

        # Initialize knots (control points)
        self.knots = None

    @staticmethod
    def _auto_nlvl(imsz, grid_spacing, k_down):
        """MATLAB formula: nlvl = min over dims of floor(log(sz/max(4,gs)) / log(1/k_down))."""
        import math
        if isinstance(grid_spacing, (list, tuple)):
            gs_per_dim = [int(g) for g in list(grid_spacing)[:len(imsz)]]
        else:
            gs_per_dim = [int(grid_spacing)] * len(imsz)
        log_k_inv = abs(math.log(k_down))  # log(1/k_down) > 0
        levels = []
        for s, gs in zip(imsz, gs_per_dim):
            ratio = float(s) / max(4, gs)
            if ratio <= 1.0:
                levels.append(0)
            else:
                levels.append(int(math.floor(math.log(ratio) / log_k_inv)))
        return max(1, min(levels))

    def _create_pyramid(self, image, levels):
        pyramid = []
        current = image

        # Gaussian blur sigma for anti-aliasing: matches MATLAB's 0.4*(0.5/k_down)
        sigma = 0.4 * 0.5 / self.k_down
        size = max(3, int(2 * np.ceil(2.0 * sigma) + 1))  # MATLAB kernel size formula
        coords = torch.arange(size, dtype=torch.float32, device=self.device) - (size - 1) / 2.0
        g1d = torch.exp(-coords**2 / (2 * sigma**2))
        g1d = g1d / g1d.sum()
        
        pad = size // 2
        
        for i in range(levels):
            pyramid.append(current)
            if i < levels - 1:
                # Anti-aliasing blur before downsampling.
                # Use depthwise (groups=C) separable Gaussian to support any number of channels.
                if current.dim() == 5:
                    C = current.shape[1]
                    kz = g1d.view(1, 1, size, 1, 1).expand(C, 1, size, 1, 1).contiguous()
                    ky = g1d.view(1, 1, 1, size, 1).expand(C, 1, 1, size, 1).contiguous()
                    kx = g1d.view(1, 1, 1, 1, size).expand(C, 1, 1, 1, size).contiguous()
                    blurred = F.conv3d(current, kz, padding=(pad, 0, 0), groups=C)
                    blurred = F.conv3d(blurred, ky, padding=(0, pad, 0), groups=C)
                    blurred = F.conv3d(blurred, kx, padding=(0, 0, pad), groups=C)
                current = F.interpolate(blurred, scale_factor=self.k_down, mode='trilinear', align_corners=True, recompute_scale_factor=True)
        return pyramid[::-1] # Coarsest to finest

    def forward(self, level_idx):
        # Current resolution
        moving_curr = self.moving_pyramid[level_idx]
        shape = moving_curr.shape[2:]
        
        # Generate dense flow from knots
        # knots: (N, 3, D_k, H_k, W_k)
        # flow: (N, 3, D, H, W)
        flow = knots_to_dense_flow(self.knots, shape)
        
        # Warp moving image
        warped = warp_image(moving_curr, flow)
        
        return warped, flow

    def _knot_shape(self, shape):
        # MATLAB-like control grid sizing: ceil(N/s) + 1 for linear splines.
        return [max(3, int(np.ceil(float(s) / float(self.grid_spacing))) + 1) for s in shape]

    def _level_pix_resolution(self, level_idx):
        # Pyramid is ordered coarsest->finest, so coarser levels have larger voxel spacing.
        scale_power = max(0, self.n_levels - 1 - level_idx)
        level_factor = self.k_down ** scale_power
        return [p / max(level_factor, 1e-8) for p in self.pix_resolution]

    def get_Dk(self, knots, level_idx):
        # Calculate gradients of knots (forward differences)
        dy = knots[:, :, :, 1:, :] - knots[:, :, :, :-1, :]
        dx = knots[:, :, :, :, 1:] - knots[:, :, :, :, :-1]
        dz = knots[:, :, 1:, :, :] - knots[:, :, :-1, :, :]
        
        # Pad to match sizes
        pad_x = F.pad(dx, (0, 1, 0, 0, 0, 0))
        pad_y = F.pad(dy, (0, 0, 0, 1, 0, 0))
        pad_z = F.pad(dz, (0, 0, 0, 0, 0, 1))
        
        # Output is (N, 3, 3, D, H, W). dim=1 is displacement component, dim=2 is diff direction
        Dk = torch.stack([pad_x, pad_y, pad_z], dim=2)
        # Knots are in pixel units.  Dimensionless strain = delta_u_px / gs,
        # because delta_u_px / gs == delta_u_mm / (pix_res * gs).
        # pix_res cancels — do NOT include pix_res in the denominator.
        return Dk / float(self.grid_spacing)

    def compute_loss(self, warped, fixed, flow, level_idx, csqrt=5e-3):
        if self.metric == 'vfc':
            # Warp the pre-computed moving VFC field with the current displacement.
            # This avoids recomputing FFT-based VFC at every L-BFGS iteration.
            mov_vfc_5d = self._cur_mov_vfc.unsqueeze(0)               # (1, 3, D, H, W)
            warped_mov_vfc = warp_image(mov_vfc_5d, flow).squeeze(0)  # (3, D, H, W)
            metric_map = self.vfc_metric.alignment_loss(
                self._cur_fix_vfc, warped_mov_vfc
            )  # (1, 1, D, H, W)
        elif fixed is not None:
            if self.metric == 'ssd':
                metric_map = (warped - fixed) ** 2
            else:
                metric_map = self.metric_fn(fixed, warped, return_map=True)
        elif self.metric == 'nuclear':
            # Groupwise Nuclear Norm Metric
            N = warped.shape[0]
            flat_images = warped.view(N, -1).T
            S = torch.linalg.svdvals(flat_images)
            metric_loss = torch.sum(S) / flat_images.shape[0]
            metric_map = None
        else:
            metric_map = None

        if metric_map is not None:
            # Create boolean active mask mapped to current scale
            active_mask = torch.ones_like(metric_map, dtype=torch.bool)

            # Apply fixed anatomical mask if provided
            if self.fixed_mask is not None:
                curr_shape = warped.shape[2:]
                mask_curr = F.interpolate(self.fixed_mask, size=curr_shape, mode='nearest') > 0.5
                active_mask = active_mask & mask_curr[:, :metric_map.shape[1], ...]

            # Border masking — scale with pyramid level (matches MATLAB behaviour)
            bm = int(round(self.border_mask * (self.k_down ** max(0, self.n_levels - 1 - level_idx))))
            if bm > 0:
                active_mask[:, :, :bm, :, :] = False
                active_mask[:, :, -bm:, :, :] = False
                active_mask[:, :, :, :bm, :] = False
                active_mask[:, :, :, -bm:, :] = False
                active_mask[:, :, :, :, :bm] = False
                active_mask[:, :, :, :, -bm:] = False

            metric_loss = metric_map[active_mask].mean() if active_mask.any() else metric_map.mean()

        # Compute smoothed Isotropic TV gradient magnitudes matching COPD_final.m
        Dk_grad = self.get_Dk(self.knots, level_idx)
        grad_mag = torch.sqrt(torch.sum(Dk_grad**2, dim=(1, 2)) + csqrt)
        tv_loss = torch.mean(grad_mag)
        
        current_lambda = self.lambda_reg[level_idx]
        return metric_loss + current_lambda * tv_loss, metric_loss, tv_loss

    def optimize(self, iterations=80, lr=1.0):
        """
        Run the multi-level optimization.
        iterations: int or list/tuple of ints. If list, specifies iterations for each level (coarsest to finest).
        """
        
        # Handle iterations argument
        if isinstance(iterations, int):
            iterations_list = [iterations] * self.n_levels
        else:
            iterations_list = list(iterations)
            # If single value in list, replicate
            if len(iterations_list) == 1:
                iterations_list = iterations_list * self.n_levels
            # If mismatch, pad with last value or truncate
            elif len(iterations_list) < self.n_levels:
                iterations_list += [iterations_list[-1]] * (self.n_levels - len(iterations_list))
            elif len(iterations_list) > self.n_levels:
                iterations_list = iterations_list[:self.n_levels]
        
        for i in range(self.n_levels):
            n_iters = iterations_list[i]
            print(f"--- Level {i} (0=coarsest, iters={n_iters}) ---")

            lvl_pix_res = self._level_pix_resolution(i)

            # Rebuild LCC metric with per-level sigma scaled from physical to pixel units
            if self.metric == 'lcc':
                # sigma_pix = sigma_mm / voxel_size_mm, clamped to >=0.8 (MATLAB behaviour)
                sigma_pix = [max(0.8, self.metric_param_mm[d] / lvl_pix_res[d]) for d in range(3)]
                self.metric_fn = LocalCrossCorrelation(sigma=sigma_pix, dim=3).to(self.device)
                print(f"  LCC sigma (pixels): {[f'{s:.2f}' for s in sigma_pix]}")

            # Rebuild VFC kernel and pre-compute both VFC fields for this level
            elif self.metric == 'vfc':
                self.vfc_metric.set_level(lvl_pix_res, self.device)
                # Fixed VFC — computed once, never changes within a level
                if self.fixed_edgemap_pyramid is not None:
                    fix_em = self.fixed_edgemap_pyramid[i].squeeze()
                    self._cur_fix_vfc = self.vfc_metric.edge_to_vfc(fix_em)
                else:
                    fix_3d = self.fixed_pyramid[i].squeeze()
                    self._cur_fix_vfc = self.vfc_metric.image_to_vfc(fix_3d)
                # Moving VFC — pre-computed from unwarped moving image and then warped
                if self.moving_edgemap_pyramid is not None:
                    mov_em = self.moving_edgemap_pyramid[i].squeeze()
                    self._cur_mov_vfc = self.vfc_metric.edge_to_vfc(mov_em)
                else:
                    mov_3d = self.moving_pyramid[i].squeeze()
                    self._cur_mov_vfc = self.vfc_metric.image_to_vfc(mov_3d)
                print(f"  VFC kernel built at pix_res={[f'{r:.2f}' for r in lvl_pix_res]} mm")

            # 1. Initialize or Upsample Knots
            current_shape = self.moving_pyramid[i].shape[2:]
            
            if self.knots is None:
                # Coarsest level initialization
                qs = self._knot_shape(current_shape)
                
                N_imgs = self.moving.shape[0]
                knot_shape = (N_imgs, 3, *qs)
                self.knots = nn.Parameter(torch.zeros(knot_shape, device=self.device))
                
            else:
                # Upsample knots from previous level
                with torch.no_grad():
                    # We compute the exact knot shape for the current pyramid level
                    qs = self._knot_shape(current_shape)
                    
                    upsampled_knots = F.interpolate(self.knots.data, size=qs, mode='trilinear', align_corners=True)
                    # Scale displacements by the inverse of the downscale factor to match the current pixel coordinate space
                    upsampled_knots = upsampled_knots * (1.0 / self.k_down) 
                    self.knots = nn.Parameter(upsampled_knots)
            
            # 2. Setup Optimizer
            optimizer = optim.LBFGS([self.knots], lr=lr, max_iter=n_iters, line_search_fn='strong_wolfe')
            
            # 3. Optimization Loop
            def closure():
                optimizer.zero_grad()
                warped, flow = self.forward(i)
                fixed_curr = self.fixed_pyramid[i] if self.fixed_pyramid else None
                loss, m_loss, tv = self.compute_loss(warped, fixed_curr, flow, i)
                # Scale loss to avoid early stopping in LBFGS due to small gradients
                scaled_loss = loss * 1e4
                scaled_loss.backward()
                return scaled_loss

            # LBFGS runs internally for multiple iterations
            for epoch in range(1):
                optimizer.step(closure)
            
            # Manually run one more forward pass strictly for logging the level
            with torch.no_grad():
                warped, flow = self.forward(i)
                fixed_curr = self.fixed_pyramid[i] if self.fixed_pyramid else None
                _, m_loss, true_tv = self.compute_loss(warped, fixed_curr, flow, i)
                print(f"Level {i}: Final Metric={m_loss.item():.6f}, True TV={true_tv.item():.6f}")
        
        return self.forward(self.n_levels - 1)
