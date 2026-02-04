import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .deformation import knots_to_dense_flow, warp_image, create_grid
from .utils import to_torch, to_numpy
from .metrics import LocalCrossCorrelation

class PTVRegistration(nn.Module):
    def __init__(self, fixed_image, moving_image, grid_spacing=4, device='cpu', 
                 lambda_reg=0.11, n_levels=3, metric='lcc', metric_param=2.1):
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

        self.grid_spacing = grid_spacing
        self.lambda_reg = lambda_reg
        self.n_levels = n_levels
        self.metric = metric
        
        if self.metric == 'lcc':
            self.metric_fn = LocalCrossCorrelation(sigma=metric_param, dim=3).to(device)
        
        self.moving_pyramid = self._create_pyramid(self.moving, n_levels)
        
        # Initialize knots (control points)
        self.knots = None 

    def _create_pyramid(self, image, levels):
        pyramid = []
        current = image
        for i in range(levels):
            pyramid.append(current)
            # Downsample for next level
            # Use trilinear for 3D
            if i < levels - 1:
                # Recompute scale factor to ensure proper downsampling
                current = F.interpolate(current, scale_factor=0.5, mode='trilinear', align_corners=True, recompute_scale_factor=True)
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

    def compute_loss(self, warped, fixed, flow):
        # Regularization (Isotropic TV on flow)
        # Calculate gradients of flow
        dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
        dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
        dz = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        
        # Pad to match sizes
        pad_x = F.pad(dx, (0, 1, 0, 0, 0, 0))
        pad_y = F.pad(dy, (0, 0, 0, 1, 0, 0))
        pad_z = F.pad(dz, (0, 0, 0, 0, 0, 1))
        
        grad_mag = torch.sqrt(pad_x**2 + pad_y**2 + pad_z**2 + 1e-6)
        tv_loss = torch.mean(grad_mag)

        if self.metric == 'ssd':
            if fixed is None:
                raise ValueError("Fixed image required for SSD metric")
            ssd_loss = torch.mean((warped - fixed) ** 2)
            metric_loss = ssd_loss
        elif self.metric == 'lcc':
            if fixed is None:
                raise ValueError("Fixed image required for LCC metric")
            metric_loss = self.metric_fn(fixed, warped)
        elif self.metric == 'nuclear':
            # Groupwise Nuclear Norm Metric
            N = warped.shape[0]
            flat_images = warped.view(N, -1).T # (Pixels, N)
            S = torch.linalg.svdvals(flat_images)
            metric_loss = torch.sum(S)
            metric_loss = metric_loss / flat_images.shape[0]

        return metric_loss + self.lambda_reg * tv_loss, metric_loss, tv_loss

    def optimize(self, iterations=80, lr=0.1):
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
                    # We simply double the resolution of the grid
                    old_knot_shape = self.knots.shape[2:]
                    new_knot_shape = [s * 2 for s in old_knot_shape]
                    
                    upsampled_knots = F.interpolate(self.knots.data, size=new_knot_shape, mode='trilinear', align_corners=True)
                    upsampled_knots = upsampled_knots * 2.0 # Scale displacements
                    self.knots = nn.Parameter(upsampled_knots)
            
            # 2. Setup Optimizer
            optimizer = optim.Adam([self.knots], lr=lr)
            
            # 3. Optimization Loop
            for it in range(n_iters):
                optimizer.zero_grad()
                warped, flow = self.forward(i)
                
                fixed_curr = self.fixed_pyramid[i] if self.fixed_pyramid else None
                loss, m_loss, tv = self.compute_loss(warped, fixed_curr, flow)
                
                loss.backward()
                optimizer.step()
                
                if it % 10 == 0:
                    print(f"Iter {it}: Loss={loss.item():.6f}, Metric={m_loss.item():.6f}, TV={tv.item():.6f}")
        
        return self.forward(self.n_levels - 1)
