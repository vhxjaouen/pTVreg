import argparse
import os
import torch
import numpy as np
import nibabel as nib
from pTVreg.deformation import warp_image_scipy, warp_image
from pTVreg.utils import save_nifti, to_torch

def main():
    parser = argparse.ArgumentParser(description='pTVwarp: Apply warp field to image')
    parser.add_argument('-i', '--input', required=True, help='Input moving image')
    parser.add_argument('-r', '--ref', required=True, help='Reference fixed image (defines space)')
    parser.add_argument('-w', '--warp', required=True, help='Warp field (NIfTI)')
    parser.add_argument('-o', '--output', required=True, help='Output warped image')
    parser.add_argument('--interp', choices=['linear', 'cubic', 'nearest'], default='cubic', 
                        help='Interpolation order (default: cubic / bsp3)')
    
    args = parser.parse_args()
    
    print(f"Loading input: {args.input}")
    input_img = nib.load(args.input)
    input_data = input_img.get_fdata()
    
    print(f"Loading reference: {args.ref}")
    ref_img = nib.load(args.ref)
    ref_shape = ref_img.shape
    
    print(f"Loading warp field: {args.warp}")
    warp_img = nib.load(args.warp)
    warp_data = warp_img.get_fdata()
    
    # Check dimensions
    # Warp field usually (D, H, W, 3) or (3, D, H, W) or (D, H, W, 1, 3)
    # Our pTVreg outputs (D, H, W, 3)
    
    if warp_data.shape[-1] == 3:
        # (D, H, W, 3) -> (3, D, H, W)
        flow = np.moveaxis(warp_data, -1, 0)
    elif warp_data.shape[0] == 3:
        flow = warp_data
    else:
        # Check for 5D (1, 1, D, H, W, 3)?
        flow = warp_data.squeeze()
        if flow.shape[-1] == 3:
            flow = np.moveaxis(flow, -1, 0)
        elif flow.shape[0] == 3:
            pass
        else:
            raise ValueError(f"Unexpected warp field shape: {warp_data.shape}")
            
    # Check if flow matches reference size
    if flow.shape[1:] != ref_shape:
        print(f"Warning: Warp field shape {flow.shape[1:]} does not match reference shape {ref_shape}. Resizing flow...")
        # We need to resize flow to match reference grid
        # Using scipy zoom or torch interpolate
        # flow is (3, D, H, W)
        import torch.nn.functional as F
        flow_t = torch.from_numpy(flow)[None] # (1, 3, D, H, W)
        new_flow = F.interpolate(flow_t, size=ref_shape, mode='trilinear', align_corners=True)
        # If we resize the grid, do we need to scale the displacements?
        # If displacements are in pixels:
        #   If I say "shift 10 pixels" in a 100x100 image.
        #   If I resize flow to 200x200, should it say "shift 20 pixels"?
        #   Usually displacement fields are resolution dependent if in pixels.
        #   pTVreg uses pixels.
        #   So yes, we should scale.
        
        scale_factors = [n / o for n, o in zip(ref_shape, flow.shape[1:])]
        # scale factors are (Sz, Sy, Sx)
        # flow channels are (dz, dy, dx)
        
        flow = new_flow[0].numpy()
        for c in range(3):
            flow[c] *= scale_factors[c]
            
    print(f"Applying warp with interpolation: {args.interp}")
    
    if args.interp == 'cubic':
        # Use scipy for cubic (order=3)
        warped = warp_image_scipy(input_data, flow, order=3)
    elif args.interp == 'linear':
        # Use scipy for linear (order=1) or torch
        # Let's stick to scipy for consistency in this script unless gpu is requested
        warped = warp_image_scipy(input_data, flow, order=1)
    elif args.interp == 'nearest':
        warped = warp_image_scipy(input_data, flow, order=0)
        
    print(f"Saving output to: {args.output}")
    save_nifti(warped, args.output, affine=ref_img.affine)

if __name__ == "__main__":
    main()
