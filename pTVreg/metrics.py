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
        
        # Outer product to get nD kernel
        kernel = kernel_1d
        for _ in range(dim - 1):
            kernel = torch.ger(kernel, kernel_1d).flatten() # This logic is slightly flawed for >2D if chaining
            
        # Correct nD construction
        if dim == 1:
            k = kernel_1d.view(1, 1, -1)
        elif dim == 2:
            k = torch.ger(kernel_1d, kernel_1d).view(1, 1, size, size)
        elif dim == 3:
            k2 = torch.ger(kernel_1d, kernel_1d)
            k = torch.mul(k2.unsqueeze(-1), kernel_1d).view(1, 1, size, size, size)
            
        return nn.Parameter(k, requires_grad=False)

    def forward(self, fixed, moving):
        # fixed, moving: (N, 1, D, H, W)
        
        # Ensure correct device
        if self.kernel.device != fixed.device:
            self.kernel = self.kernel.to(fixed.device)
            
        # Convolutions
        # We perform local mean calculation
        # mu_I = I * G
        
        # Padding to keep size same
        padding = self.kernel_size // 2
        
        def conv(input_img):
            if self.dim == 3:
                return F.conv3d(input_img, self.kernel, padding=padding)
            elif self.dim == 2:
                return F.conv2d(input_img, self.kernel, padding=padding)
        
        # Compute local sums (means)
        I = fixed
        J = moving
        
        I2 = I * I
        J2 = J * J
        IJ = I * J
        
        mu_I = conv(I)
        mu_J = conv(J)
        mu_I2 = conv(I2)
        mu_J2 = conv(J2)
        mu_IJ = conv(IJ)
        
        # Variances and Covariance
        # var_I = E[I^2] - (E[I])^2
        var_I = mu_I2 - mu_I**2
        var_J = mu_J2 - mu_J**2
        cov_IJ = mu_IJ - mu_I * mu_J
        
        # Zero variance check (epsilon)
        epsilon = 1e-5
        
        # LCC = cov^2 / (var_I * var_J)
        cc = cov_IJ**2 / (var_I * var_J + epsilon)
        
        # Return negative mean CC (minimize)
        return -torch.mean(cc)
