import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .deformation import knots_to_dense_flow, warp_image
from .utils import to_torch, to_numpy
from .metrics import LocalCrossCorrelation

class PTVRegistration(nn.Module):
    def __init__(self, fixed_image, moving_image, grid_spacing=4, device='cpu', 
                 lambda_reg=0.11, n_levels=3, metric='lcc', metric_param=2.1, k_down=0.7):
        """
        Args:
            fixed_image: (D, H, W) or (N, D, H, W) or None (for groupwise)
            moving_image: (D, H, W) or (N, D, H, W)
            grid_spacing: int, spacing of control points at finest level
            lambda_reg: regularization weight (default 0.11 for COPD)
            n_levels: number of pyramid levels
            metric: 'ssd', 'nuclear', or 'lcc' (Local Cross Correlation)
            metric_param: parameter for metric (e.g. sigma for LCC)
        """
        super().__init__()
        self.device = device
        self.grid_spacing = grid_spacing
        self.lambda_reg = lambda_reg
        self.n_levels = n_levels
        self.metric = metric
        self.k_down = k_down
        
        # Handle inputs
        self.moving = to_torch(moving_image, device=device)
        if self.moving.ndim == 3:
            self.moving = self.moving[None, None, ...] # (1, 1, D, H, W)
        elif self.moving.ndim == 4:
            self.moving = self.moving[:, None, ...] # (N, 1, D, H, W)
            
        if fixed_image is not None:
            self.fixed = to_torch(fixed_image, device=device)
            if self.fixed.ndim == 3:
                self.fixed = self.fixed[None, None, ...]
            elif self.fixed.ndim == 4:
                self.fixed = self.fixed[:, None, ...]
            self.fixed_pyramid = self._create_pyramid(self.fixed, n_levels)
        else:
            self.fixed = None
            self.fixed_pyramid = None

        if self.metric == 'lcc':
            self.metric_fn = LocalCrossCorrelation(sigma=metric_param, dim=3).to(device)
        
        self.moving_pyramid = self._create_pyramid(self.moving, n_levels)
        
        # Initialize knots (control points)
        self.knots = None 

    def _create_pyramid(self, image, levels):
        pyramid = []
        current = image
        
        # Pre-create a simple Gaussian blur kernel for anti-aliasing during downsampling
        # Sigma roughly matched to downscale factor 0.7 for standard anti-aliasing
        sigma = 0.5 / self.k_down 
        size = int(4 * sigma + 1) | 1 # ensure odd
        coords = torch.arange(size, dtype=torch.float32, device=self.device) - (size - 1) / 2.0
        g1d = torch.exp(-coords**2 / (2 * sigma**2))
        g1d = g1d / g1d.sum()
        
        kz = g1d.view(1, 1, size, 1, 1)
        ky = g1d.view(1, 1, 1, size, 1)
        kx = g1d.view(1, 1, 1, 1, size)
        
        pad = size // 2
        
        for i in range(levels):
            pyramid.append(current)
            if i < levels - 1:
                # Anti-aliasing blur before downsampling
                if current.dim() == 5:
                    blurred = F.conv3d(current, kz, padding=(pad, 0, 0))
                    blurred = F.conv3d(blurred, ky, padding=(0, pad, 0))
                    blurred = F.conv3d(blurred, kx, padding=(0, 0, pad))
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

    def get_Dk(self, knots):
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
        # Scale to dimensionless (pixel diff / pixel dist)
        return Dk / max(1.0, float(self.grid_spacing))

    def compute_loss(self, warped, fixed, flow, csqrt=5e-3):
        if self.metric == 'ssd':
            if fixed is None:
                raise ValueError("Fixed image required for SSD metric")
            metric_loss = torch.mean((warped - fixed) ** 2)
        elif self.metric == 'lcc':
            if fixed is None:
                raise ValueError("Fixed image required for LCC metric")
            metric_loss = self.metric_fn(fixed, warped)
        elif self.metric == 'nuclear':
            # Groupwise Nuclear Norm Metric
            N = warped.shape[0]
            flat_images = warped.view(N, -1).T # (Pixels, N)
            S = torch.linalg.svdvals(flat_images)
            metric_loss = torch.sum(S) / flat_images.shape[0]

        # Compute smoothed Isotropic TV gradient magnitudes matching COPD_final.m
        Dk_grad = self.get_Dk(self.knots)
        grad_mag = torch.sqrt(torch.sum(Dk_grad**2, dim=(1, 2)) + csqrt)
        tv_loss = torch.mean(grad_mag)
        
        return metric_loss + self.lambda_reg * tv_loss, metric_loss, tv_loss

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
            
            # 1. Initialize or Upsample Knots
            current_shape = self.moving_pyramid[i].shape[2:]
            
            if self.knots is None:
                # Coarsest level initialization
                qs = [max(3, s // self.grid_spacing) for s in current_shape]
                
                N_imgs = self.moving.shape[0]
                knot_shape = (N_imgs, 3, *qs)
                self.knots = nn.Parameter(torch.zeros(knot_shape, device=self.device))
                
            else:
                # Upsample knots from previous level
                with torch.no_grad():
                    # We compute the exact knot shape for the current pyramid level
                    qs = [max(3, s // self.grid_spacing) for s in current_shape]
                    
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
                loss, m_loss, tv = self.compute_loss(warped, fixed_curr, flow)
                loss.backward()
                return loss

            # LBFGS runs internally for multiple iterations
            for epoch in range(1):
                optimizer.step(closure)
            
            # Manually run one more forward pass strictly for logging the level
            with torch.no_grad():
                warped, flow = self.forward(i)
                fixed_curr = self.fixed_pyramid[i] if self.fixed_pyramid else None
                _, m_loss, true_tv = self.compute_loss(warped, fixed_curr, flow)
                print(f"Level {i}: Final Metric={m_loss.item():.6f}, True TV={true_tv.item():.6f}")
        
        return self.forward(self.n_levels - 1)
