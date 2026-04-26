import argparse
import sys
import os
import json
import torch
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from pTVreg.registration import PTVRegistration
from pTVreg.utils import save_nifti
from pTVreg.deformation import warp_image


def main():
    parser = argparse.ArgumentParser(description='pTVreg: Python Image Registration', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', help='Path to JSON configuration file to bypass command line arguments')
    parser.add_argument('-f', '--fixed', help='Fixed image path')
    parser.add_argument('-m', '--moving', help='Moving image path')
    parser.add_argument('--fixed_mask', default=None, help='Path to fixed image mask (metric only evaluated where mask > 0)')
    parser.add_argument('--mask_labels', type=int, nargs='+', default=None, help='List of integers to extract from the multi-label mask (e.g., --mask_labels 1 2). If not provided, all non-zero values are used.')
    parser.add_argument('--border_mask', type=int, default=5, help='Number of voxels to ignore at the image boundaries to prevent padding artifacts')
    parser.add_argument('--clip', type=float, nargs=2, default=None, help='Clip intensities to [min, max] before normalizing (e.g., 80 900 for lung CT)')
    parser.add_argument('-o', '--output', default='./moving_warped.nii.gz', help='Output warped image path')
    
    parser.add_argument('-w', '--warp', nargs='?', const='./warpfield.nii.gz', default=None, 
                        help='Output warp field path. If flag is set without value, defaults to ./warpfield.nii.gz')
    
    parser.add_argument('--gpu', action='store_true', help='Force use GPU if available')
    parser.add_argument('--spacing', type=int, default=8, help='Grid spacing at finest level, in voxels (indicative: 4, 8, 16)')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Downsample scale factor (e.g. 0.5 for 50%% size) to save memory and speed up processing.')
    
    parser.add_argument('--lambda_reg', type=float, nargs='+', default=[0.15], help='Regularization weight base, or list per scale. If a single value is provided, it is applied uniformly across all scales. (indicative range: 0.05 - 0.5)')
    parser.add_argument('--iterations', type=int, nargs='+', default=[160, 120, 80], help='Iterations per level. Can be a single scalar or a list (coarsest to finest). (indicative: 50-300 per level)')
    parser.add_argument('--metric', choices=['lcc', 'ssd', 'nuclear', 'emse', 'ngf', 'vfc'], default='lcc', help='Similarity metric (default: lcc)')
    parser.add_argument('--metric_param', type=float, default=2.1, help='Metric parameter (e.g. LCC sigma, NGF alpha). (indicative ranges: 1.5 - 4.0 for LCC, )')

    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    if args.config:
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(args, key) and key != 'config':
                    setattr(args, key, value)
                    
    if not args.fixed or not args.moving:
        parser.error('the following arguments are required: -f/--fixed, -m/--moving (unless --config is provided)')
    
    print(f"Loading fixed: {args.fixed}")
    fixed_img = nib.load(args.fixed)
    fixed_data = fixed_img.get_fdata().astype(np.float32)
    
    print(f"Loading moving: {args.moving}")
    moving_img = nib.load(args.moving)
    moving_data = moving_img.get_fdata().astype(np.float32)
    
    fixed_mask = None
    if args.fixed_mask:
        print(f"Loading fixed mask: {args.fixed_mask}")
        mask_img = nib.load(args.fixed_mask)
        mask_data = mask_img.get_fdata()
        
        if args.mask_labels is not None:
            print(f"Filtering mask for labels: {args.mask_labels}")
            fixed_mask = np.isin(mask_data, args.mask_labels).astype(np.float32)
        else:
            fixed_mask = (mask_data > 0).astype(np.float32)
        
    if args.clip:
        cl_min, cl_max = args.clip
        print(f"Clipping intensities to [{cl_min}, {cl_max}]")
        fixed_data = np.clip(fixed_data, cl_min, cl_max)
        moving_data = np.clip(moving_data, cl_min, cl_max)
        
    orig_shape = fixed_data.shape

    # Normalize intensities strictly to [0, 1] 
    # Ensures LCC correlation scales predictably within float32 limits
    fixed_data = (fixed_data - np.min(fixed_data)) / (np.ptp(fixed_data) + 1e-8)
    moving_min = np.min(moving_data)
    moving_ptp = np.ptp(moving_data) + 1e-8
    moving_data = (moving_data - moving_min) / moving_ptp

    if args.scale_factor != 1.0:
        print(f"Scaling images by {args.scale_factor} for memory optimization...")
        # Scale for faster convergence and lower memory imprint on massive volumes
        fixed_data = ndimage.zoom(fixed_data, args.scale_factor, order=1)
        moving_data = ndimage.zoom(moving_data, args.scale_factor, order=1)
        print(f"Scaled shapes: {fixed_data.shape}")

    
    
    # Auto-detect GPU
    if torch.cuda.is_available():
        device = 'cuda'
        print("GPU detected and used by default.")
    else:
        device = 'cpu'
        print("GPU not available, using CPU.")
    
    print(f"Using device: {device}")
    
    n_levels = len(args.iterations) if len(args.iterations) > 1 else 3
    
    reg = PTVRegistration(fixed_data, moving_data, 
                          fixed_mask=fixed_mask,
                          border_mask=args.border_mask,
                          grid_spacing=args.spacing, 
                          device=device,
                          lambda_reg=args.lambda_reg, 
                          n_levels=n_levels,
                          metric=args.metric,
                          metric_param=args.metric_param)
    
    print(f"Starting optimization (Metric={args.metric}, Reg={args.lambda_reg}, Iters={args.iterations})...")
    # Pass the iterations list/scalar to optimize
    warped, flow = reg.optimize(iterations=args.iterations)
    
    # Save warped image
    print(f"Saving warped image to: {args.output}")

    # Normalize intensities strictly to [0, 1] 
    # Ensures LCC correlation scales predictably within float32 limits
    fixed_data = (fixed_data - np.min(fixed_data)) / (np.ptp(fixed_data) + 1e-8)
    moving_min = np.min(moving_data)
    moving_ptp = np.ptp(moving_data) + 1e-8
    moving_data = (moving_data - moving_min) / moving_ptp

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
        warped_np = (warped_np * moving_ptp) + moving_min
        flow_np = full_flow.detach().cpu().numpy()
    else:
        warped_np = warped.detach().cpu().numpy().squeeze()
        
        flow_np = flow.detach().cpu().numpy().squeeze() # (3, D, H, W)
        
    # Use affine from fixed image since we registered to it
    
    # Identify optimal dtype to save disk space
    target_dtype = moving_img.get_data_dtype()
    if np.issubdtype(target_dtype, np.floating):
        # Force float32 to prevent 64-bit bloat (MRIs don't need 64-bit precision)
        warped_np = warped_np.astype(np.float32)
        # Drop precision to ~10 bits mantissa (float16) to drastically improve GZIP compression without visual loss
        warped_np = warped_np.astype(np.float16).astype(np.float32)
    else:
        # Round before casting to integer to avoid truncation artifacts
        warped_np = np.round(warped_np).astype(target_dtype)

    save_nifti(warped_np, args.output, affine=fixed_img.affine)
    
    # Save warp field if requested
    if args.warp:
        print(f"Saving warp field to: {args.warp}")
        
        # Save as (D, H, W, 3) for compatibility with standard viewers
        flow_save = np.moveaxis(flow_np, 0, -1)
        
        # Compress warp field powerfully using fixed-point int16 quantization
        # A scale of 0.01 provides 1/100th of a voxel precision (more than enough for medical fields)
        # while taking up exactly half the raw space and compressing massively due to dropped floating-point noise.
        scale = 0.01
        
        # Clip just in case, though deformations > 327 voxels are absurd.
        flow_quantized = np.clip(np.round(flow_save / scale), -32768, 32767).astype(np.int16)
        
        warp_img = nib.Nifti1Image(flow_quantized, fixed_img.affine)
        warp_img.header.set_data_dtype(np.int16)
        warp_img.header['scl_slope'] = scale
        warp_img.header['scl_inter'] = 0.0
        
        nib.save(warp_img, args.warp)

    # Save registration metadata to a JSON sidecar file
    json_path = args.output.replace('.nii.gz', '.json').replace('.nii', '.json')
    print(f"Saving registration metadata to: {json_path}")

    # Filter out or convert non-serializable arguments if needed
    metadata = vars(args).copy()
    # Add actual device used (cpu/cuda)
    metadata['device_used'] = device
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    main()
