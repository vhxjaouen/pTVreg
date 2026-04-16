import re

with open("pTVreg/metrics.py", "r") as f:
    text = f.read()

addition = """
from kornia.filters import SpatialGradient3d

class NormalizedGradientField(nn.Module):
    def __init__(self, alpha_NGF=0.05):
        super().__init__()
        self.alpha_NGF = alpha_NGF
        self.spatial_grad = SpatialGradient3d()
        
    def forward(self, fixed, moving):
        # fixed, moving: (N, 1, D, H, W)
        
        grad_src = self.spatial_grad(fixed)
        grad_tgt = self.spatial_grad(moving)
        
        # kornia SpatialGradient3d returns (B, C, 3, D, H, W)
        src_x = grad_src[:, :, 0, ...]
        src_y = grad_src[:, :, 1, ...]
        src_z = grad_src[:, :, 2, ...]
        
        tgt_x = grad_tgt[:, :, 0, ...]
        tgt_y = grad_tgt[:, :, 1, ...]
        tgt_z = grad_tgt[:, :, 2, ...]
        
        gradmag_src = torch.sqrt(src_x**2 + src_y**2 + src_z**2 + self.alpha_NGF**2)
        gradmag_tgt = torch.sqrt(tgt_x**2 + tgt_y**2 + tgt_z**2 + self.alpha_NGF**2)
        
        eps = 1e-8
        NGF = 1.0 - 0.5 * (((src_x / (gradmag_src + eps)) * (tgt_x / (gradmag_tgt + eps)) + 
                            (src_y / (gradmag_src + eps)) * (tgt_y / (gradmag_tgt + eps)) + 
                            (src_z / (gradmag_src + eps)) * (tgt_z / (gradmag_tgt + eps))) ** 2)
        
        return torch.mean(NGF)
        
"""

text += addition

with open("pTVreg/metrics.py", "w") as f:
    f.write(text)
