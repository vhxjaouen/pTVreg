import re

with open('pTVreg/cli.py', 'r') as f:
    content = f.read()

# Replace the argparse setup
content = re.sub(
    r"parser.add_argument\('-o', '--output', default='./moving_warped.nii.gz', help='Output warped image path'\)",
    "parser.add_argument('-o', '--output', default='./moving_warped.nii.gz', help='Output warped image path')\n    parser.add_argument('-e', '--edge', nargs='*', type=float, default=None, help='Align gradient edge magnitude instead of images. Optional: two floats for gaussian smoothing sigmas (fixed, moving).')",
    content
)

# Replace the first normalization
old_norm = """    orig_shape = fixed_data.shape

    # Normalize intensities strictly to [0, 1] 
    # Ensures LCC correlation scales predictably within float32 limits
    fixed_data = (fixed_data - np.min(fixed_data)) / (np.ptp(fixed_data) + 1e-8)
    moving_min = np.min(moving_data)
    moving_ptp = np.ptp(moving_data) + 1e-8
    moving_data = (moving_data - moving_min) / moving_ptp"""

new_norm = """    orig_shape = fixed_data.shape

    if args.edge is not None:
        print("Computing gradient edge magnitudes...")
        sig_fixed = args.edge[0] if len(args.edge) >= 1 else 0.0
        sig_moving = args.edge[1] if len(args.edge) >= 2 else 0.0
        
        def compute_edge(data, sigma):
            if sigma > 0.0:
                return ndimage.gaussian_gradient_magnitude(data, sigma)
            else:
                grads = np.gradient(data)
                return np.sqrt(sum(g**2 for g in grads))
                
        fixed_data = compute_edge(fixed_data, sig_fixed)
        moving_data = compute_edge(moving_data, sig_moving)

    # Normalize intensities strictly to [0, 1] 
    # Ensures LCC correlation scales predictably within float32 limits
    fixed_data = (fixed_data - np.min(fixed_data)) / (np.ptp(fixed_data) + 1e-8)
    moving_min = np.min(moving_data)
    moving_ptp = np.ptp(moving_data) + 1e-8
    moving_data = (moving_data - moving_min) / moving_ptp"""

content = content.replace(old_norm, new_norm)

# Replace the messy warped output part entirely
old_warp_start = "    # Normalize intensities strictly to [0, 1]"
old_warp_end = "        flow_np = flow.detach().cpu().numpy().squeeze() # (3, D, H, W)"

# Find start and end exactly manually
import copy
lines = content.split('\n')
idx_start = -1
idx_end = -1
for i, line in enumerate(lines):
    if line.strip() == "# Save warped image":
        idx_start = i + 2 # Skip print line
    if line.strip() == "flow_np = flow.detach().cpu().numpy().squeeze() # (3, D, H, W)":
        idx_end = i + 1

if idx_start > 0 and idx_end > 0:
    new_warp_lines = """
    if flow.dim() == len(orig_shape) + 1:
        flow = flow.unsqueeze(0)

    if args.scale_factor != 1.0:
        print(f"Upsampling warp field back to original size {orig_shape}...")
        mode = 'trilinear' if len(orig_shape) == 3 else 'bilinear'
        full_flow = torch.nn.functional.interpolate(flow, size=orig_shape, mode=mode, align_corners=True).squeeze(0)
        full_flow = full_flow / args.scale_factor
    else:
        full_flow = flow.squeeze(0)

    print("Warping original resolution moving image with flow field...")
    moving_orig_tensor = torch.tensor(moving_img.get_fdata().astype(np.float32), device=device).unsqueeze(0).unsqueeze(0)
    full_flow_tensor = full_flow.unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            warped_np = warp_image(moving_orig_tensor, full_flow_tensor).squeeze().cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        print("VRAM limit exceeded during final full-resolution WARP! Falling back to CPU warp...")
        moving_orig_tensor = moving_orig_tensor.cpu()
        full_flow_tensor = full_flow_tensor.cpu()
        with torch.no_grad():
            warped_np = warp_image(moving_orig_tensor, full_flow_tensor).squeeze().numpy()

    flow_np = full_flow.detach().cpu().numpy()
"""
    new_lines = lines[:idx_start] + new_warp_lines.strip('\n').split('\n') + lines[idx_end:]
    content = '\n'.join(new_lines)


with open('pTVreg/cli.py', 'w') as f:
    f.write(content)
print("Done patching.")
