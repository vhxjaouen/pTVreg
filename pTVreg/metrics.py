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

    def forward(self, fixed, moving, return_map=False):
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
        
        # Cache static correlation values for the fixed image.
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
        
        # Zero variance regularization epsilon (1e-8 for normalized [0,1] images).
        epsilon = 1e-8
        
        # LCC calculation: cov^2 / (var_I * var_J)
        cc = cov_IJ**2 / (var_I * var_J + epsilon)
        
        # Return negative mean local cross-correlation.
        return -cc if return_map else -torch.mean(cc)

from kornia.filters import SpatialGradient3d

class EdgeMSE(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, img1, img2, return_map=False):
        grad_src = SpatialGradient3d()(img1)
        grad_tgt = SpatialGradient3d()(img2)

        src_x = grad_src[:,:,0,:,:,:]
        src_y = grad_src[:,:,1,:,:,:]
        src_z = grad_src[:,:,2,:,:,:]
        tgt_x = grad_tgt[:,:,0,:,:,:]
        tgt_y = grad_tgt[:,:,1,:,:,:]
        tgt_z = grad_tgt[:,:,2,:,:,:]

        gradmag_src = torch.sqrt(torch.pow(src_x, 2) + torch.pow(src_y, 2) + torch.pow(src_z, 2) + 1e-8)
        gradmag_tgt = torch.sqrt(torch.pow(tgt_x, 2) + torch.pow(tgt_y, 2) + torch.pow(tgt_z, 2) + 1e-8)
        
        # Mean Squared Error on the edge magnitudes
        emse = (gradmag_src - gradmag_tgt)**2
        # Normalize the MSE by the variance/mean of the fixed image's edge magnitude
        # This keeps the gradients significantly active and bounded on the order of 1.0
        # otherwise the sub-0.001 gradients are completely masked by the lambda_reg penalty
        emse_norm = emse / (torch.mean(gradmag_src**2) + 1e-8)
        if return_map: return emse_norm
        return torch.mean(emse_norm)

class NormalizedGradientField(nn.Module):
    def __init__(self, alpha_NGF=0.05):
        super().__init__()
        self.alpha_NGF = alpha_NGF
        
    def forward(self, img1, img2, return_map=False):
        grad_src = SpatialGradient3d()(img1)
        grad_tgt = SpatialGradient3d()(img2)

        src_x = grad_src[:,:,0,:,:,:]
        src_y = grad_src[:,:,1,:,:,:]
        src_z = grad_src[:,:,2,:,:,:]
        tgt_x = grad_tgt[:,:,0,:,:,:]
        tgt_y = grad_tgt[:,:,1,:,:,:]
        tgt_z = grad_tgt[:,:,2,:,:,:]

        mag_src = torch.sqrt(src_x**2 + src_y**2 + src_z**2 + 1e-8)
        mag_tgt = torch.sqrt(tgt_x**2 + tgt_y**2 + tgt_z**2 + 1e-8)

        # Adaptive alpha: correct the vanishing gradient by scaling the noise threshold 
        # relative to the actual mean gradient magnitude of the images.
        # This properly weights the vectors regardless of [0, 1] normalizations.
        alpha_src_sq = (self.alpha_NGF * torch.mean(mag_src))**2 + 1e-8
        alpha_tgt_sq = (self.alpha_NGF * torch.mean(mag_tgt))**2 + 1e-8

        gradmag_src = torch.sqrt(mag_src**2 + alpha_src_sq)
        gradmag_tgt = torch.sqrt(mag_tgt**2 + alpha_tgt_sq)
        
        eps = 1e-8
        dot_sq = torch.pow((src_x/(gradmag_src + eps) * tgt_x/(gradmag_tgt + eps) + 
                            src_y/(gradmag_src + eps) * tgt_y/(gradmag_tgt + eps) + 
                            src_z/(gradmag_src + eps) * tgt_z/(gradmag_tgt + eps)), 2)
        
        NGF = 1.0 - 0.5 * dot_sq

        # To ensure the solver gradients compete robustly with the TV regularization multiplier (~0.15),
        # we can scale the loss strictly to a zero-centered landscape, or just return the standard mean.
        # The adaptive alpha natively restores the gradient magnitudes!
        if return_map:
            return NGF * 100.0
        else:
            return torch.mean(NGF) * 100.0

class VectorFieldConvolution(nn.Module):
    def __init__(self, gamma=1.7, kernel_size=11):
        super().__init__()
        self.gamma = gamma
        self.kernel_size = kernel_size
        
        # Create Vector Field kernel
        self._create_vfc_kernel()
        
    def _create_vfc_kernel(self):
        # Create grid centered at 0
        r = self.kernel_size // 2
        z, y, x = torch.meshgrid(
            torch.arange(-r, r + 1, dtype=torch.float32),
            torch.arange(-r, r + 1, dtype=torch.float32),
            torch.arange(-r, r + 1, dtype=torch.float32),
            indexing='ij'
        )
        
        # Calculate distance R
        dist = torch.sqrt(x**2 + y**2 + z**2 + 1e-8)
        
        # Vector field components pointing towards the edges
        vx = -x / (dist ** (self.gamma + 1) + 1e-8)
        vy = -y / (dist ** (self.gamma + 1) + 1e-8)
        vz = -z / (dist ** (self.gamma + 1) + 1e-8)
        
        # Origin has 0 influence
        vx[dist < 1e-8] = 0
        vy[dist < 1e-8] = 0
        vz[dist < 1e-8] = 0
        
        # Reshape to (Out_channels, In_channels, D, H, W) -> (3, 1, D, H, W)
        kernel = torch.stack([vx, vy, vz], dim=0).unsqueeze(1)
        self.register_buffer('vfc_kernel', kernel)
        
    def forward(self, img1, img2, return_map=False):
        grad_src = SpatialGradient3d()(img1)
        grad_tgt = SpatialGradient3d()(img2)

        src_x = grad_src[:,:,0,:,:,:]
        src_y = grad_src[:,:,1,:,:,:]
        src_z = grad_src[:,:,2,:,:,:]
        tgt_x = grad_tgt[:,:,0,:,:,:]
        tgt_y = grad_tgt[:,:,1,:,:,:]
        tgt_z = grad_tgt[:,:,2,:,:,:]

        # Edge magnitude maps
        mag_src = torch.sqrt(src_x**2 + src_y**2 + src_z**2 + 1e-8)
        mag_tgt = torch.sqrt(tgt_x**2 + tgt_y**2 + tgt_z**2 + 1e-8)

        # 1. Scale edge maps strictly to [0,1] range per sample
        b = mag_src.shape[0]
        max_src = mag_src.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1, 1) + 1e-8
        max_tgt = mag_tgt.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1, 1) + 1e-8
        
        mag_src = mag_src / max_src
        mag_tgt = mag_tgt / max_tgt

        # Compute Dense Vector Convolution Fields 
        # Output shape: (N, 3, D, H, W)
        padding = self.kernel_size // 2
        vfc_src = F.conv3d(mag_src, self.vfc_kernel, padding=padding)
        vfc_tgt = F.conv3d(mag_tgt, self.vfc_kernel, padding=padding)
        
        # 2. Maximize the dot product (Cosine Similarity) of the vector fields!
        # Rather than flattening MSE, we measure structural alignment directly.
        # The adaptive alpha threshold suppresses chaotic gradients in empty background zones.
        alpha_sq = torch.mean(vfc_src**2) + 1e-8
        
        dot = torch.sum(vfc_src * vfc_tgt, dim=1, keepdim=True)
        mag_sq_src = torch.sum(vfc_src**2, dim=1, keepdim=True) + alpha_sq
        mag_sq_tgt = torch.sum(vfc_tgt**2, dim=1, keepdim=True) + alpha_sq
        
        # vfc_ncc in [-1, 1]
        vfc_ncc = dot / torch.sqrt(mag_sq_src * mag_sq_tgt)
        
        # We negate it to maximize the dot product alignment, scaling to compete with lambda_reg (0.15)
        return -vfc_ncc * 100.0 if return_map else -torch.mean(vfc_ncc) * 100.0