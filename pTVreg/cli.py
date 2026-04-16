import argparse
import sys
import os
import torch
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from pTVreg.registration import PTVRegistration
from pTVreg.utils import save_nifti
from pTVreg.deformation import warp_image


def main():
    parser = argparse.ArgumentParser(description='pTVreg: Python Image Registration', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--fixed', required=True, help='Fixed image path')
    parser.add_argument('-m', '--moving', required=True, help='Moving image path')
    parser.add_argument('-o', '--output', default='./moving_warped.nii.gz', help='Output warped image path')
    
    parser.add_argument('-w', '--warp', nargs='?', const='./warpfield.nii.gz', default=None, 
                        help='Output warp field path. If flag is set without value, defaults to ./warpfield.nii.gz')
    
    parser.add_argument('--gpu', action='store_true', help='Force use GPU if available')
    parser.add_argument('--levels', type=int, default=3, help='Number of pyramid levels (indicative range: 3-5)')
    parser.add_argument('--spacing', type=int, default=8, help='Grid spacing at finest level, in voxels (indicative: 4, 8, 16)')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Downsample scale factor (e.g. 0.5 for 50%% size) to save memory and speed up processing.')
    
    parser.add_argument('--lambda_reg', type=float, nargs='+', default=[0.15], help='Regularization weight base, or list per scale. If a single value is provided, it is used for the finest scale, and multiplied by 2 for each coarser scale (indicative range: 0.05 - 0.5)')
    parser.add_argument('--iterations', type=int, nargs='+', default=[160, 120, 80], help='Iterations per level. Can be a single scalar or a list (coarsest to finest). (indicative: 50-300 per level)')
    parser.add_argument('--metric', choices=['lcc', 'ssd', 'nuclear'], default='lcc', help='Similarity metric (default: lcc)')
    parser.add_argument('--metric_param', type=float, default=2.1, help='Metric parameter (e.g. LCC sigma). (indicative range: 1.5 - 4.0)')

    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    print(f"Loading fixed: {args.fixed}")
    fixed_img = nib.load(args.fixed)
    fixed_data = fixed_img.get_fdata().astype(np.float32)
    
    print(f"Loading moving: {args.moving}")
    moving_img = nib.load(args.moving)
    moving_data = moving_img.get_fdata().astype(np.float32)
    
    orig_shape = fixed_data.shape
    if args.scale_factor != 1.0:
        print(f"Scaling images by {args.scale_factor} for memory optimization...")
        # Scale for faster convergence and lower memory imprint on massive volumes
        fixed_data = ndimage.zoom(fixed_data, args.scale_factor, order=1)
        moving_data = ndimage.zoom(moving_data, args.scale_factor, order=1)
        print(f"Scaled shapes: {fixed_data.shape}")

    # Normalize intensities strictly to [0, 1] 
    # Ensures LCC correlation scales predictably within float32 limits
    fixed_data = (fixed_data - np.min(fixed_data)) / (np.ptp(fixed_data) + 1e-8)
    moving_min = np.min(moving_data)
    moving_ptp = np.ptp(moving_data) + 1e-8
    moving_data = (moving_data - moving_min) / moving_ptp
    
    # Auto-detect GPU
    if torch.cuda.is_available():
        device = 'cuda'
        print("GPU detected and used by default.")
    else:
        device = 'cpu'
        print("GPU not available, using CPU.")
    
    print(f"Using device: {device}")
    
    reg = PTVRegistration(fixed_data, moving_data, 
                          grid_spacing=args.spacing, 
                          device=device,
                          lambda_reg=args.lambda_reg, 
                          n_levels=args.levels,
                          metric=args.metric,
                          metric_param=args.metric_param)
    
    print(f"Starting optimization (Metric={args.metric}, Reg={args.lambda_reg}, Iters={args.iterations})...")
    # Pass the iterations list/scalar to optimize
    warped, flow = reg.optimize(iterations=args.iterations)
    
    # Save warped image
    print(f"Saving warped image to: {args.output}")
    if args.scale_factor != 1.0:
        # Upsample flow back to original shape: (1, 3, D, H, W)
        print(f"Upsampling warp field back to original size {orig_shape}...")
        
        # Scale displacement magnitudes by inverse of scale factor
        if flow.dim() == len(orig_shape) + 1:
            flow = flow.unsqueeze(0)
        mode = 'trilinear' if len(orig_shape) == 3 else 'bilinear'
        full_flow = torch.nn.functional.interpolate(flow, size=orig_shape, mode=mode, align_corners=True).squeeze(0)
        full_flow = full_flow / args.scale_factor
        
        # Target original image space resolution to prevent quality loss in output
        print("Warping original resolution moving image with upsampled flow field...")
        moving_orig_tensor = torch.tensor(moving_img.get_fdata().astype(np.float32), device=device).unsqueeze(0).unsqueeze(0)
        full_flow_tensor = full_flow.unsqueeze(0).to(device)
        
        # If running large volumes, dynamically fallback to host RAM if GPU memory gets exceeded on the final high-res grid sample
        try:
            with torch.no_grad():
                original_warped = warp_image(moving_orig_tensor, full_flow_tensor).squeeze().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            print("VRAM limit exceeded during final full-resolution WARP! Falling back to CPU warp...")
            moving_orig_tensor = moving_orig_tensor.cpu()
            full_flow_tensor = full_flow_tensor.cpu()
            with torch.no_grad():
                original_warped = warp_image(moving_orig_tensor, full_flow_tensor).squeeze().numpy()
            
        warped_np = original_warped
        flow_np = full_flow.detach().cpu().numpy()
    else:
        warped_np = warped.detach().cpu().numpy().squeeze()
        warped_np = (warped_np * moving_ptp) + moving_min
        flow_np = flow.detach().cpu().numpy().squeeze() # (3, D, H, W)
        
    # Use affine from fixed image since we registered to it
    warped_np = warped_np.astype(moving_img.get_data_dtype())
    save_nifti(warped_np, args.output, affine=fixed_img.affine)
    
    # Save warp field if requested
    if args.warp:
        print(f"Saving warp field to: {args.warp}")
        
        # Save as (D, H, W, 3) for compatibility with standard viewers
        # Compress as float16 to save significant disk space (deformations are usually smooth, precision loss is negligible)
        flow_save = np.moveaxis(flow_np, 0, -1).astype(np.float16) 

        
        # Save vector field
        # We use the same affine as fixed image
        save_nifti(flow_save, args.warp, affine=fixed_img.affine)

if __name__ == "__main__":
    main()
