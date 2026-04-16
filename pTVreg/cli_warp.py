import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

def save_nifti(data, path, affine=np.eye(4)):
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)

def main():
    parser = argparse.ArgumentParser(description='pTVwarp: Apply warp field to moving image, resampling to reference space.')
    parser.add_argument('-i', '--input', required=True, help='Input moving image')
    parser.add_argument('-r', '--ref', required=True, help='Reference fixed image (defines output space and header)')
    parser.add_argument('-w', '--warp', required=True, help='Warp field (NIfTI, pTVreg format)')
    parser.add_argument('-o', '--output', required=True, help='Output warped image')
    parser.add_argument('--interp', choices=['linear', 'cubic', 'nearest'], default='cubic', 
                        help='Interpolation order (default: cubic)')
    
    args = parser.parse_args()
    
    print(f"Loading reference: {args.ref}")
    ref_img = nib.load(args.ref)
    ref_data = ref_img.get_fdata()
    ref_shape = ref_data.shape
    ref_affine = ref_img.affine
    
    print(f"Loading moving input: {args.input}")
    input_img = nib.load(args.input)
    input_data = input_img.get_fdata()
    input_affine = input_img.affine
    
    print(f"Loading warp field: {args.warp}")
    warp_img = nib.load(args.warp)
    warp_data = warp_img.get_fdata()
    
    # Process flow shape: pTVreg warp is usually (D, H, W, 3)
    if warp_data.shape[-1] == 3:
        flow = np.moveaxis(warp_data, -1, 0)
    elif warp_data.shape[0] == 3:
        flow = warp_data
    else:
        flow = warp_data.squeeze()
        if flow.shape[-1] == 3:
            flow = np.moveaxis(flow, -1, 0)
        elif flow.shape[0] == 3:
            pass
        else:
            raise ValueError(f"Unexpected warp field shape: {warp_data.shape}")

    # Check and resize flow if it doesn't match the reference shape
    if flow.shape[1:] != ref_shape:
        print(f"Warning: Warp field shape {flow.shape[1:]} does not match reference shape {ref_shape}. Resizing flow...")
        from scipy.ndimage import zoom
        scale_factors = [n / o for n, o in zip(ref_shape, flow.shape[1:])]
        zoom_factors = [1.0] + scale_factors 
        flow = zoom(flow, zoom_factors, order=1)
        # Scale flow displacements accordingly
        for c in range(3):
            flow[c] *= scale_factors[c]

    # Create voxel grid for reference image
    ndim = len(ref_shape)
    coords_ref = np.indices(ref_shape, dtype=np.float32)
    
    # Add flow (which is in reference voxel space). Flow channels match dimensions (e.g., z, y, x)
    coords_ref_warped = coords_ref + flow
    
    # Flatten to list of coordinates for affine transformation
    # ref coords shape: (3, N)
    coords_ref_warped_flat = coords_ref_warped.reshape(ndim, -1)
    
    # Augment to homogeneous coordinates (4, N)
    coords_ref_warped_hom = np.vstack([coords_ref_warped_flat, np.ones((1, coords_ref_warped_flat.shape[1]))])
    
    # M_ref -> physical
    phys_coords = ref_affine @ coords_ref_warped_hom
    
    # physical -> M_mov
    inv_input_affine = np.linalg.inv(input_affine)
    mov_coords_hom = inv_input_affine @ phys_coords
    
    # Extract moving voxel coordinates (3, N)
    mov_coords = mov_coords_hom[:ndim, :]
    
    # Reshape back to (3, D, H, W)
    mov_coords = mov_coords.reshape((ndim,) + ref_shape)
    
    print(f"Applying warp with interpolation: {args.interp}")
    order_map = {'nearest': 0, 'linear': 1, 'cubic': 3}
    
    warped = map_coordinates(input_data, mov_coords, order=order_map[args.interp], mode='nearest')
    
    print(f"Saving output to: {args.output}")
    save_nifti(warped, args.output, affine=ref_affine)
    print("Done.")

if __name__ == "__main__":
    main()
