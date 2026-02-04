# pTVreg Python Port

This directory contains a pure Python/PyTorch implementation of the pTVreg image registration algorithms, originally written in MATLAB/MEX.

## Features

*   **GPU Acceleration**: Uses PyTorch for all operations, enabling massive speedups on CUDA devices.
*   **Automatic Differentiation**: No need for manual gradient derivations; PyTorch handles backpropagation through the objective function.
*   **Multi-Resolution Pyramid**: Robust registration using coarse-to-fine optimization.
*   **Metrics**:
    *   SSD (Sum of Squared Differences) for pairwise registration.
    *   Nuclear Norm for groupwise registration (minimizing rank of joint image matrix).
*   **Regularization**: Isotropic Total Variation (TV) on the displacement field.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Pairwise Registration (Fixed vs Moving)

```python
from pTVreg_python.registration import PTVRegistration
import nibabel as nib

# Load data
fixed = nib.load('fixed.nii.gz').get_fdata()
moving = nib.load('moving.nii.gz').get_fdata()

# Initialize
reg = PTVRegistration(fixed, moving, grid_spacing=4, device='cuda', lambda_reg=0.1)

# Optimize
warped, flow = reg.optimize()

# Save
nib.save(nib.Nifti1Image(warped.cpu().numpy(), np.eye(4)), 'warped.nii.gz')
```

### Groupwise Registration

```python
# Load multiple images as a batch (N, D, H, W)
moving_images = ... 

# Initialize with 'nuclear' metric and no fixed image
reg = PTVRegistration(None, moving_images, metric='nuclear', ...)

warped_images, flows = reg.optimize()
```

## Files

*   `registration.py`: Main `PTVRegistration` class handling the optimization loop and pyramid.
*   `deformation.py`: Core geometric primitives (B-Spline grid upsampling, dense flow generation, image warping). Replaces the C++ MEX files.
*   `utils.py`: I/O helpers.

## Comparison to MATLAB

The original MATLAB code used custom C++ MEX files (`imdeform`, `linear_disp`, etc.) for performance. This port replaces them with `torch.nn.functional.grid_sample` and `interpolate`, which are highly optimized and GPU-ready.
