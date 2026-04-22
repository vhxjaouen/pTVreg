# pTVreg Python Port

This directory contains a pure Python/PyTorch implementation of the pTVreg image registration algorithms, originally written in MATLAB/MEX (`COPD_final.m`, `ptv_register.m`). This port has been extensively optimized to match and exceed the original C++ performance while solving critical mathematical rigidities in PyTorch.

## New Features & Optimizations

*   **GPU Acceleration**: Uses PyTorch for all operations, enabling massive speedups on CUDA devices.
*   **Exact MATLAB Parity**: Matches the `csqrt=5e-3` Huber-smoothed $L_2/L_1$ optimization scheme on B-Spline knot control points using L-BFGS to stop unrealistic biological organ stretching.
*   **Massive Speedups**:
    *   **Separable 1D Convolutions**: Dense $O(N^3)$ 3D Gaussians for the Local Cross-Correlation (LCC) metric mathematically factored into sequential $O(3N)$ 1D boundaries.
    *   **Grid Caching**: Zero-overhead spatial coordinate generation via `_cached_grids` for instantaneous `F.grid_sample` lookups mapping pixels to indices.
    *   **Anti-aliasing**: Strict sequential 1D Gaussian blurring precedes all multi-resolution pyramid decimation steps.
*   **Memory Efficiency for Large Volumes ([512, 512, 531])**:
    *   `--scale_factor` argument allows dynamic downsampling before computing the dense multi-level gradient optimization to fit 24GB VRAM limits.
    *   Automatically upsamples the resultant warp field via `trilinear` interpolation and applies the final warp cleanly at original resolution (dropping to CPU if VRAM bursts).
*   **Intensity Normalization**: Actively normalizes all input images strictly to `[0, 1]` ranges and pads zero-variance denominators with `1e-8` to prevent L-BFGS line-search explosions and parameter dilution.

## Installation

You can install the package and its CLI tool globally or in your virtual environment:

```bash
# Install as an editable package (recommended for development)
pip install -e .
```

*Note: This automatically installs all required dependencies from `pyproject.toml` and makes the `ptvreg` command available everywhere.*

## Usage

### Command Line Interface (CLI)

The package provides two CLI tools:

### 1. `ptvreg` (Image Registration)

The `ptvreg` CLI single-handedly handles scaling, solver optimization, and final high-res warp application.

```bash
ptvreg -f ct.nii.gz -m mri.nii.gz -w warp.nii.gz -o warped.nii.gz \
    --spacing 8 \
    --lambda_reg 0.15 \
    --iterations 160 120 80 \
    --scale_factor 0.5 
```

**Parameters Check:**
*   `--spacing 8`: Allows smooth, non-tearing biological boundaries in 3D patches.
*   `--lambda_reg 0.15`: The target regularization weight (coupled with the internal 5e-3 Huber norm smoothing). If a single value is provided (like 0.15), it is applied to the *finest* resolution scale. For each coarser scale above it, the weight is multiplied by 2 (e.g., for 3 levels, it becomes `[0.6, 0.3, 0.15]`). This enforces stiffer, smoother deformations at coarse levels to prevent large-scale folding, while allowing more precise sub-pixel alignments at fine levels. If you want explicit manual control over every scale, you can pass multiple values (e.g., `--lambda_reg 0.5 0.2 0.05`).
*   `--iterations 160 120 80`: Drives the solver deeper on the coarsest scale, resolving gross mismatch before handling fine details.
*   `--scale_factor 0.5`: Down-scales the entire volumetric stack (e.g. `512x512x531` -> `256x256x265`), solves the L-BFGS, then accurately reconstructs a `512` flow field.
*   `--metric emse`: Selects the metric to optimize. Available metrics include `lcc` (Local Cross-Correlation), `ssd` (Sum of Squared Differences), `nuclear`, AND `emse` (Edge Mean Squared Error), and `ngf` (Normalized Gradient Field).
*   `--metric_param 0.05`: Used to adjust the respective selected metric dynamically (e.g. LCC requires a variance Gaussian $\sigma$ like `2.1`.

### 2. `ptvwarp` (Apply Transformation)

Use `ptvwarp` to apply a previously computed transformation field onto a new image (e.g., reslicing moving segmented masks into the fixed image physical space natively via exact NIfTI affines without registration):

```bash
ptvwarp -i moving_mask.nii.gz -r fixed_ct.nii.gz -w warp.nii.gz -o resliced_mask.nii.gz --interp nearest
```

### Pairwise Registration (Fixed vs Moving in Code)

```python
from pTVreg.registration import PTVRegistration
import nibabel as nib

# Load data
fixed = nib.load('fixed.nii.gz').get_fdata()
moving = nib.load('moving.nii.gz').get_fdata()

# Initialize
reg = PTVRegistration(fixed, moving, grid_spacing=8, device='cuda', lambda_reg=0.15)

# Optimize
warped, flow = reg.optimize(iterations=[160, 120, 80])

# Do what you wish with warped array
```

## Files

*   `pTVreg/cli.py`: The single CLI entry point that replaces previously separated scripts (`pTVreg.py`/`pTVwarp.py`). Handles PyTorch GPU dispatch, Niftis, and dynamic VRAM fallback scaling.
*   `pTVreg/registration.py`: Main `PTVRegistration` class handling the optimization loop, gradients and multi-resolution Gaussians.
*   `pTVreg/deformation.py`: Core geometric primitives. Grid scaling, caching, displacement tensor math.
*   `pTVreg/metrics.py`: The heavily structured LCC, SSD, and TV mathematical constraints.
