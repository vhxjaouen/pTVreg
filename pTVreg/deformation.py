import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import map_coordinates

_cached_grids = {}

def create_normalized_grid(shape, device='cpu'):
    """
    Create a [-1, 1] normalized coordinate grid with (x,y,z) order,
    compatible with F.grid_sample.
    """
    key = (tuple(shape), str(device))
    if key in _cached_grids:
        return _cached_grids[key]
        
    ranges = [torch.linspace(-1.0, 1.0, steps=s, device=device) for s in shape]
    # 'ij' indexing creates ranges matching dimensions (e.g. z, y, x)
    grid = torch.meshgrid(*ranges, indexing='ij')
    
    # grid_sample expects coordinates ordered identically to dimensions but flipped to (x,y,z)
    if len(shape) == 3:
        grid_xyz = torch.stack([grid[2], grid[1], grid[0]], dim=-1)
    else:
        grid_xyz = torch.stack([grid[1], grid[0]], dim=-1)
    
    grid_xyz = grid_xyz.unsqueeze(0) # (1, D, H, W, 3)
    
    _cached_grids[key] = grid_xyz
    return grid_xyz

def knots_to_dense_flow(knots, output_shape, mode='trilinear'):
    """
    Upsample the sparse control point grid (knots) to a dense flow field.
    """
    return F.interpolate(knots, size=output_shape, mode=mode, align_corners=True)

def warp_image(image, flow, mode='bilinear', padding_mode='border'):
    """
    Warp an image using a dense displacement field (flow).
    Args:
        image: (N, C, D, H, W) image tensor
        flow: (N, 3, D, H, W) displacement field in PIXELS (order: z, y, x for 3D)
    """
    shape = image.shape[2:]
    norm_grid = create_normalized_grid(shape, device=image.device)
    
    # Scale flow from pixels to normalized [-1, 1] offsets
    # z, y, x order in sizes matching shape[0], shape[1], shape[2]
    scale = [2.0 / (s - 1.0) for s in shape]
    scale_tensor = torch.tensor(scale, device=flow.device, dtype=flow.dtype).view(1, len(shape), *((1,) * len(shape)))
    norm_flow = flow * scale_tensor
    
    # Repackage flow from (z, y, x) channels to the (x, y, z) last-dimension format
    norm_flow_permuted = norm_flow.flip(dims=[1]).permute(0, 2, 3, 4, 1)
    
    sample_coords = norm_grid + norm_flow_permuted
    
    return F.grid_sample(image, sample_coords, mode=mode, padding_mode=padding_mode, align_corners=True)

def warp_image_scipy(image, flow, order=3, mode='nearest'):
    shape = image.shape
    coords = np.indices(shape, dtype=np.float32)
    map_coords = coords + flow
    warped = map_coordinates(image, map_coords, order=order, mode=mode)
    return warped
