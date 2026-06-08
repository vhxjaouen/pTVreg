import argparse
import sys
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import map_coordinates

def save_nifti(data, path, affine=np.eye(4)):
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)

def main():
    parser = argparse.ArgumentParser(description='pTVwarp: Apply warp field to moving image, resampling to reference space.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, help='Input moving image')
    parser.add_argument('-r', '--ref', required=True, help='Reference fixed image (defines output space and header)')
    parser.add_argument('-w', '--warp', required=True, help='Warp field (NIfTI, pTVreg format)')
    parser.add_argument('-o', '--output', required=True, help='Output warped image')
    parser.add_argument('--interp', choices=['linear', 'cubic', 'nearest'], default='cubic', 
                        help='Interpolation order (default: cubic)')
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    print(f"Loading reference: {args.ref}")
    ref_img = nib.load(args.ref)
    ref_data = ref_img.get_fdata()
    # Use only spatial (first 3) dimensions as the output grid
    ref_shape = tuple(ref_data.shape[:3])
    ndim = 3
    ref_affine = ref_img.affine
    
    print(f"Loading moving input: {args.input}")
    input_img = nib.load(args.input)
    input_data = input_img.get_fdata()
    input_affine = input_img.affine
    
    print(f"Loading warp field: {args.warp}")
    warp_img = nib.load(args.warp)
    warp_data = warp_img.get_fdata()
    
    # Process flow shape: pTVreg warp is (D, H, W, 3) or (3, D, H, W)
    if warp_data.ndim == 4 and warp_data.shape[-1] == 3:
        flow = np.moveaxis(warp_data, -1, 0)   # -> (3, D, H, W)
    elif warp_data.ndim == 4 and warp_data.shape[0] == 3:
        flow = warp_data
    else:
        warp_sq = warp_data.squeeze()
        if warp_sq.ndim == 4 and warp_sq.shape[-1] == 3:
            flow = np.moveaxis(warp_sq, -1, 0)
        elif warp_sq.ndim == 4 and warp_sq.shape[0] == 3:
            flow = warp_sq
        else:
            raise ValueError(f"Unexpected warp field shape: {warp_data.shape}")

    # Reslice flow to reference spatial shape if sizes differ
    flow_shape = tuple(flow.shape[1:])
    if flow_shape != ref_shape:
        print(f"Reslicing warp field from {flow_shape} to {ref_shape}...")
        scale_factors = [n / o for n, o in zip(ref_shape, flow_shape)]
        # Use torch for efficient trilinear upsampling
        flow_t = torch.from_numpy(flow).unsqueeze(0).float()  # (1, 3, d, h, w)
        flow_t = torch.nn.functional.interpolate(
            flow_t, size=ref_shape, mode='trilinear', align_corners=True
        ).squeeze(0)  # (3, D, H, W)
        # Scale displacement magnitudes to match the new voxel grid
        for c in range(3):
            flow_t[c] *= scale_factors[c]
        flow = flow_t.numpy()

    # Create voxel grid for reference image
    coords_ref = np.indices(ref_shape, dtype=np.float32)  # (3, D, H, W)
    
    # Add flow (displacements in reference voxel units, order: z, y, x)
    coords_ref_warped = coords_ref + flow
    
    # Flatten to (3, N), augment to homogeneous (4, N)
    coords_ref_warped_flat = coords_ref_warped.reshape(ndim, -1)
    coords_ref_warped_hom = np.vstack([coords_ref_warped_flat, np.ones((1, coords_ref_warped_flat.shape[1]))])
    
    # ref voxel -> physical -> moving voxel
    phys_coords = ref_affine @ coords_ref_warped_hom
    inv_input_affine = np.linalg.inv(input_affine)
    mov_coords_hom = inv_input_affine @ phys_coords
    
    # Extract and reshape moving voxel coordinates to (3, D, H, W)
    mov_coords = mov_coords_hom[:ndim, :].reshape((ndim,) + ref_shape)
    
    print(f"Applying warp with interpolation: {args.interp}")
    order_map = {'nearest': 0, 'linear': 1, 'cubic': 3}
    
    warped = map_coordinates(input_data, mov_coords, order=order_map[args.interp], mode='nearest')
    
    print(f"Saving output to: {args.output}")
    save_nifti(warped, args.output, affine=ref_affine)
    print("Done.")

if __name__ == "__main__":
    main()

