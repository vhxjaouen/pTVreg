import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import map_coordinates

def create_grid(shape, device='cpu'):
    """
    Create a standard grid of coordinates for the given shape.
    Args:
        shape: (D, H, W) or (H, W)
        device: torch device
    Returns:
        grid: (1, D, H, W, 3) or (1, H, W, 2) tensor of coordinates
    """
    # Create normalized grid in [-1, 1] for grid_sample
    # But for displacement fields in pixels, we might want pixel coordinates.
    # pTVreg uses pixel displacements.
    
    # However, PyTorch grid_sample expects normalized coordinates [-1, 1].
    # We will work with normalized coordinates for the warping implementation to match PyTorch conventions,
    # or we handle the conversion internally.
    
    # Let's generate a meshgrid of pixel coordinates first.
    ranges = [torch.arange(s, device=device, dtype=torch.float32) for s in shape]
    grid = torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grid, dim=-1) # (D, H, W, 3)
    grid = grid.unsqueeze(0) # (1, D, H, W, 3)
    return grid

def knots_to_dense_flow(knots, output_shape, mode='trilinear'):
    """
    Upsample the sparse control point grid (knots) to a dense flow field.
    Args:
        knots: (N, 3, D_k, H_k, W_k) tensor of control point displacements
        output_shape: (D, H, W) tuple of output spatial dimensions
    Returns:
        flow: (N, 3, D, H, W) dense displacement field
    """
    # align_corners=True matches the behavior where the corner control points map to the corner pixels
    # This usually makes sense for registration grids.
    return F.interpolate(knots, size=output_shape, mode=mode, align_corners=True)

def warp_image(image, flow, mode='bilinear', padding_mode='border'):
    """
    Warp an image using a dense displacement field (flow).
    Args:
        image: (N, C, D, H, W) image tensor
        flow: (N, 3, D, H, W) displacement field in PIXELS (order: z, y, x for 3D)
        mode: interpolation mode ('bilinear', 'nearest')
        padding_mode: 'zeros', 'border', 'reflection'
    Returns:
        warped_image: (N, C, D, H, W)
    """
    shape = image.shape[2:]
    
    # Generate identity grid
    grid = create_grid(shape, device=image.device) # (1, D, H, W, 3)
    
    # Permute flow to match grid shape: (N, 3, D, H, W) -> (N, D, H, W, 3)
    flow_permuted = flow.permute(0, 2, 3, 4, 1)
    
    # Add displacement to grid
    # grid is (z, y, x)
    sample_coords = grid + flow_permuted
    
    # Normalize coordinates to [-1, 1] for grid_sample
    # 2*x / (W-1) - 1
    # We need to normalize each dimension independently
    
    # shape is (D, H, W)
    # sample_coords is ..., (z, y, x)
    
    norm_coords = sample_coords.clone()
    for i in range(len(shape)):
        # Normalize the i-th coordinate (z, y, x) using the i-th dimension size
        norm_coords[..., i] = 2.0 * sample_coords[..., i] / (shape[i] - 1.0) - 1.0
        
    # PyTorch grid_sample expects (x, y, z) order for the last dimension, NOT (z, y, x)
    # This is a common pitfall. 
    # 5D input (N, C, D, H, W) -> grid (N, D, H, W, 3) where last dim is (x, y, z)
    
    # Our grid generation was 'ij' indexing -> (z, y, x).
    # So we need to flip the last dimension of norm_coords before passing to grid_sample.
    norm_coords = torch.flip(norm_coords, dims=[-1])
    
    return F.grid_sample(image, norm_coords, mode=mode, padding_mode=padding_mode, align_corners=True)

def warp_image_scipy(image, flow, order=3, mode='nearest'):
    """
    Warp an image using scipy for high-order interpolation (e.g. cubic).
    Args:
        image: (D, H, W) numpy array
        flow: (3, D, H, W) numpy array, displacement in pixels (z, y, x)
        order: interpolation order (0=nearest, 1=linear, 3=cubic)
        mode: padding mode ('nearest', 'reflect', 'constant', 'wrap')
    Returns:
        warped: (D, H, W) numpy array
    """
    # Create coordinate grid
    shape = image.shape
    # indices: (3, D, H, W) -> z, y, x grids
    coords = np.indices(shape, dtype=np.float32)
    
    # Add displacement
    map_coords = coords + flow
    
    # Warp
    # map_coordinates expects coords as (ndim, output_shape...)
    warped = map_coordinates(image, map_coords, order=order, mode=mode)
    return warped
