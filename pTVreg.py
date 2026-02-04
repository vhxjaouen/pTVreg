import argparse
import os
import torch
import numpy as np
import nibabel as nib
from pTVreg.registration import PTVRegistration
from pTVreg.utils import save_nifti

def main():
    parser = argparse.ArgumentParser(description='pTVreg: Python Image Registration')
    parser.add_argument('-f', '--fixed', required=True, help='Fixed image path')
    parser.add_argument('-m', '--moving', required=True, help='Moving image path')
    parser.add_argument('-o', '--output', default='./moving_warped.nii.gz', help='Output warped image path (default: ./moving_warped.nii.gz)')
    
    # -w arg: None if not present. If present but no value, use './warpfield.nii.gz'. If present with value, use value.
    parser.add_argument('-w', '--warp', nargs='?', const='./warpfield.nii.gz', default=None, 
                        help='Output warp field path. If flag is set without value, defaults to ./warpfield.nii.gz')
    
    parser.add_argument('--gpu', action='store_true', help='Force use GPU if available (default: auto-detect)')
    parser.add_argument('--levels', type=int, default=3, help='Number of pyramid levels')
    parser.add_argument('--spacing', type=int, default=4, help='Grid spacing at finest level')
    
    # COPD defaults
    parser.add_argument('--lambda_reg', type=float, default=0.11, help='Regularization weight (default: 0.11)')
    parser.add_argument('--iterations', type=int, nargs='+', default=[80], help='Iterations per level. Can be a single scalar (default: 80) or a list (coarsest to finest).')
    parser.add_argument('--metric', choices=['lcc', 'ssd', 'nuclear'], default='lcc', help='Similarity metric (default: lcc)')
    parser.add_argument('--metric_param', type=float, default=2.1, help='Metric parameter (e.g. LCC sigma, default: 2.1)')
    
    args = parser.parse_args()
    
    print(f"Loading fixed: {args.fixed}")
    fixed_img = nib.load(args.fixed)
    fixed_data = fixed_img.get_fdata()
    
    print(f"Loading moving: {args.moving}")
    moving_img = nib.load(args.moving)
    moving_data = moving_img.get_fdata()
    
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
    warped_np = warped.detach().cpu().numpy().squeeze()
    # Use affine from fixed image since we registered to it
    save_nifti(warped_np, args.output, affine=fixed_img.affine)
    
    # Save warp field if requested
    if args.warp:
        print(f"Saving warp field to: {args.warp}")
        flow_np = flow.detach().cpu().numpy().squeeze() # (3, D, H, W)
        
        # Save as (D, H, W, 3) for compatibility with standard viewers
        flow_save = np.moveaxis(flow_np, 0, -1) 
        
        # Save vector field
        # We use the same affine as fixed image
        save_nifti(flow_save, args.warp, affine=fixed_img.affine)

if __name__ == "__main__":
    main()
