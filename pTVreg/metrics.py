import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LocalCrossCorrelation(nn.Module):
    def __init__(self, sigma=2.1, kernel_size=None, dim=3, padding_mode='reflect'):
        super().__init__()
        self.sigma = sigma
        self.dim = dim
        self.padding_mode = padding_mode
        
        if kernel_size is None:
            # 2*ceil(2*sigma) + 1 is a common rule, or 4*sigma
            kernel_size = int(4 * sigma + 0.5)
            if kernel_size % 2 == 0:
                kernel_size += 1
        self.kernel_size = kernel_size
        
        # Create Gaussian Kernel
        self.kernel = self._create_gaussian_kernel(sigma, kernel_size, dim)

    def _create_gaussian_kernel(self, sigma, size, dim):
        # Create 1D Gaussian
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        kernel_1d = torch.exp(-coords**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Save separable 1D kernels to optimize 3D operations (O(N^3) -> O(3N))
        self.kernel_1d = kernel_1d
        if dim == 2:
            k_y = kernel_1d.view(1, 1, size, 1)
            k_x = kernel_1d.view(1, 1, 1, size)
            return nn.ParameterList([nn.Parameter(k_y, requires_grad=False), nn.Parameter(k_x, requires_grad=False)])
        elif dim == 3:
            k_z = kernel_1d.view(1, 1, size, 1, 1)
            k_y = kernel_1d.view(1, 1, 1, size, 1)
            k_x = kernel_1d.view(1, 1, 1, 1, size)
            return nn.ParameterList([
                nn.Parameter(k_z, requires_grad=False), 
                nn.Parameter(k_y, requires_grad=False), 
                nn.Parameter(k_x, requires_grad=False)
            ])
            
        return None

    def forward(self, fixed, moving):
        # fixed, moving: (N, 1, D, H, W)
        
        # Ensure correct device
        if self.kernel[0].device != fixed.device:
            for i in range(len(self.kernel)):
                self.kernel[i] = self.kernel[i].to(fixed.device)
            
        # Convolutions
        # We perform local mean calculation using mathematically equivalent separable 1D Convs
        padding = self.kernel_size // 2
        
        def conv(input_img):
            x = input_img
            if self.dim == 3:
                x = F.conv3d(x, self.kernel[0], padding=(padding, 0, 0))
                x = F.conv3d(x, self.kernel[1], padding=(0, padding, 0))
                x = F.conv3d(x, self.kernel[2], padding=(0, 0, padding))
            elif self.dim == 2:
                x = F.conv2d(x, self.kernel[0], padding=(padding, 0))
                x = F.conv2d(x, self.kernel[1], padding=(0, padding))
            return x

        # Compute local sums (means)
        I = fixed
        J = moving
        
        # Cache static correlation values for the fixed image 
        # (Massive 40% speed up assuming fixed image doesn't change throughout L-BFGS loop)
        if not hasattr(self, '_cached_fixed') or self._cached_fixed is not I:
            self._cached_fixed = I
            self._mu_I = conv(I)
            self._mu_I2 = conv(I * I)
            
        mu_I = self._mu_I
        mu_I2 = self._mu_I2
        
        J2 = J * J
        IJ = I * J
        
        mu_J = conv(J)
        mu_J2 = conv(J2)
        mu_IJ = conv(IJ)
        
        # Variances and Covariance
        # var_I = E[I^2] - (E[I])^2
        # Clip to ensure valid float variances under single-precision
        var_I = F.relu(mu_I2 - mu_I**2)
        var_J = F.relu(mu_J2 - mu_J**2)
        cov_IJ = mu_IJ - mu_I * mu_J
        
        # Zero variance check (epsilon)
        # Scaled down to 1e-8 since images are normalized to [0,1]
        # Soft tissue variations might only have var ~ 1e-3, meaning var_I * var_J ~ 1e-6.
        # An epsilon of 1e-5 would completely flatten the gradient topology.
        epsilon = 1e-8
        
        # LCC = cov^2 / (var_I * var_J)
        cc = cov_IJ**2 / (var_I * var_J + epsilon)
        
        # Return negative mean CC (minimize)
        return -torch.mean(cc)
