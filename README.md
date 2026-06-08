# pTVreg Python Port

A pure Python/PyTorch implementation of the pTVreg deformable image registration algorithm, faithfully ported from the original MATLAB/MEX code (`ptv_register.m`, `COPD_final.m`) and validated to match MATLAB output quality on clinical CT data.

## MATLAB Parity

Five critical bugs present in earlier Python ports have been identified and fixed:

| # | Component | Bug | Fix |
|---|-----------|-----|-----|
| 1 | Pyramid | Hardcoded `n_levels=3`; MATLAB computes ~9 levels automatically | `_auto_nlvl()`: `min(floor(log(sz/max(4,gs)) / log(1/k_down)))` over all dims |
| 2 | LCC sigma | Fixed at 2.1 px for all levels and all axes | Per-level: `sigma_px = max(0.8, sigma_mm / voxel_size_mm)` per axis |
| 3 | Pyramid blur | `sigma = 0.5/k_down` — 2.5× too large | `sigma = 0.4 * 0.5 / k_down` matching MATLAB |
| 4 | LCC formula | Used $r^2$ (squared correlation) | Uses Pearson $r$, loss = $1 - r$, variance stabilised with `deps=1e-5` before sqrt |
| 5 | TV units | Divided by `pix_res × grid_spacing` (mm²/voxel), making z-TV ~9× weaker on anisotropic CT | `Dk / grid_spacing` only — pix_res cancels because strain = $\Delta u_\text{px} / gs$ |

## Features

- **GPU acceleration** via PyTorch; automatic CPU fallback on VRAM overflow
- **Auto pyramid levels** matching MATLAB's formula (typically 9 levels for 512³ CT)
- **Anisotropic LCC** — sigma scaled independently per axis from physical mm spacing
- **Separable 1-D Gaussian** anti-aliasing before each pyramid decimation step
- **Huber-smoothed isotropic TV** regularisation on B-spline knot displacements (`csqrt=5e-3`)
- **L-BFGS with strong Wolfe line search** — same solver as MATLAB
- **VFC metric** (Vector Field Convolution, Li & Acton 2007) — self-contained, no extra dependencies; uses kornia for edge extraction when available
- **`--scale_factor`** pre-downsampling for volumes too large to fit in GPU memory
- **`--clip`** intensity windowing before normalisation (e.g. `-1000 400` for lung CT)

## Installation

```bash
pip install -e .
```

This installs all dependencies from `pyproject.toml` and makes `ptvreg` and `ptvwarp` available on `PATH`.

## Usage

### `ptvreg` — registration

#### Quick start (LCC, matching MATLAB `COPD_final.m` settings)

```bash
ptvreg \
  -f fixed.nii.gz \
  -m moving.nii.gz \
  -o moving_warped.nii.gz \
  -w warpfield.nii.gz \
  --spacing 4 --metric lcc --metric_param 2.1 \
  --lambda_reg 0.11 --border_mask 5 --iterations 80 --gpu
```

#### VFC metric

```bash
ptvreg \
  -f fixed.nii.gz \
  -m moving.nii.gz \
  -o moving_warped.nii.gz \
  -w warpfield.nii.gz \
  --spacing 4 --metric vfc --vfc-radius 15.0 --vfc-gamma 1.7 \
  --lambda_reg 0.11 --border_mask 5 --iterations 80 --gpu
```

Supply pre-computed edge maps (e.g. from label boundaries) instead of deriving them from image gradients:

```bash
ptvreg \
  -f fixed.nii.gz -m moving.nii.gz -o out.nii.gz \
  --metric vfc \
  --fixed-edgemap fixed_edges.nii.gz \
  --moving-edgemap moving_edges.nii.gz \
  --vfc-sign-invariant
```

#### JSON config file

```bash
ptvreg -c config.json
```

#### Full argument reference

| Argument | Default | Description |
|----------|---------|-------------|
| `-c, --config` | — | JSON config file; bypasses all other CLI arguments |
| `-f, --fixed` | — | Fixed (target) image (NIfTI) |
| `-m, --moving` | — | Moving (source) image (NIfTI) |
| `-o, --output` | `./moving_warped.nii.gz` | Output warped image |
| `-w, --warp` | `./warpfield.nii.gz` | Output warp field (omit flag to skip saving) |
| `--fixed_mask` | — | Fixed-image mask; metric evaluated only where mask > 0 |
| `--mask_labels` | — | Label IDs to extract from a multi-label mask |
| `--border_mask` | `5` | Voxels to exclude at image borders |
| `--clip` | — | Intensity window `[min max]` before normalisation (e.g. `-1000 400`) |
| `--gpu` | off | Use CUDA if available |
| `--spacing` | `8` | Control-point grid spacing at finest level (voxels); try 4 for high-res CT |
| `--scale_factor` | `1.0` | Pre-downsample factor to reduce memory (e.g. `0.5`) |
| `--lambda_reg` | `0.15` | TV regularisation weight; single value or list per level (coarsest→finest) |
| `--iterations` | `160 120 80` | L-BFGS iterations per level; single value or list |
| `--metric` | `lcc` | Similarity metric: `lcc`, `ssd`, `nuclear`, `emse`, `vfc` |
| `--metric_param` | `2.1` | LCC sigma in mm |
| `--fixed-edgemap` | — | Pre-computed edge-magnitude NIfTI for fixed image (VFC only) |
| `--moving-edgemap` | — | Pre-computed edge-magnitude NIfTI for moving image (VFC only) |
| `--vfc-radius` | `15.0` | VFC kernel radius in mm — larger = smoother, longer-range pull |
| `--vfc-gamma` | `1.7` | VFC kernel decay exponent — larger = more localised |
| `--vfc-sign-invariant` | off | Use $\cos^2$ alignment loss (good for label-derived edge maps) |

---

### `ptvwarp` — apply a saved warp field

```bash
ptvwarp \
  -i moving_mask.nii.gz \
  -r fixed.nii.gz \
  -w warpfield.nii.gz \
  -o resliced_mask.nii.gz \
  --interp nearest
```

| Argument | Description |
|----------|-------------|
| `-i, --input` | Moving image to warp |
| `-r, --ref` | Reference (fixed) image — defines output space |
| `-w, --warp` | Warp field saved by `ptvreg` |
| `-o, --output` | Output image |
| `--interp` | `linear` \| `cubic` \| `nearest` (use `nearest` for segmentations) |

---

### Python API

```python
from pTVreg.registration import PTVRegistration
import nibabel as nib

fixed  = nib.load('fixed.nii.gz').get_fdata().astype('float32')
moving = nib.load('moving.nii.gz').get_fdata().astype('float32')

reg = PTVRegistration(
    fixed, moving,
    grid_spacing=4,
    pix_resolution=[1.5, 1.5, 2.5],   # mm, from NIfTI header
    device='cuda',
    metric='lcc',
    metric_param=2.1,
    lambda_reg=0.11,
    border_mask=5,
    n_levels=None,                     # auto-compute à la MATLAB
)
warped, flow = reg.optimize(iterations=80)
```

#### VFC via Python API

```python
reg = PTVRegistration(
    fixed, moving,
    grid_spacing=4,
    pix_resolution=[1.5, 1.5, 2.5],
    device='cuda',
    metric='vfc',
    vfc_radius=15.0,
    vfc_gamma=1.7,
    lambda_reg=0.11,
    # Optionally pass pre-computed edge maps (numpy arrays, same shape as images):
    # fixed_edgemap=fixed_edges,
    # moving_edgemap=moving_edges,
    # vfc_sign_invariant=True,
)
warped, flow = reg.optimize(iterations=80)
```

## VFC metric — how it works

Vector Field Convolution (Li & Acton, *IEEE Trans. Image Process.* 2007) replaces pointwise intensity comparison with a comparison of smooth *attraction fields* computed from image edges.

1. An edge-magnitude map $e(\mathbf{x}) \in [0,1]$ is extracted from each image (kornia `SpatialGradient3d` when available, otherwise central differences in mm units).
2. The VFC kernel $\mathbf{k}(\mathbf{r}) = -\hat{\mathbf{r}} / \|\mathbf{r}\|^{\gamma+1}$ is built once per pyramid level in physical (mm) space, then normalised per component.
3. Each edge map is convolved with $\mathbf{k}$ via zero-padded FFT, producing a smooth 3-component vector field $\mathbf{F} \in \mathbb{R}^{3 \times D \times H \times W}$ that points toward nearby edges.
4. The fixed VFC field is pre-computed once per level. The moving VFC field is also pre-computed once, then **warped** (not recomputed) at each L-BFGS iteration — this is ~50× cheaper than rerunning the FFT convolution per step.
5. The loss is the mean **cosine distance**: $\mathcal{L} = 1 - \hat{\mathbf{F}}_\text{fix} \cdot \hat{\mathbf{F}}_\text{mov,warped}$

With `--vfc-sign-invariant` the loss becomes $1 - \cos^2(\cdot)$, which is insensitive to sign flips — useful when edge maps are derived from label boundaries.

## File overview

| File | Purpose |
|------|---------|
| `pTVreg/registration.py` | `PTVRegistration` class: multi-scale L-BFGS optimisation loop, pyramid construction, TV regularisation |
| `pTVreg/metrics.py` | `LocalCrossCorrelation`, `VFCMetric`, `EdgeMSE`, `MultiChannelLCC` |
| `pTVreg/deformation.py` | Grid sampling, displacement field primitives, `warp_image` |
| `pTVreg/cli.py` | `ptvreg` CLI entry point |
| `pTVreg/cli_warp.py` | `ptvwarp` CLI entry point |
| `pTVreg/utils.py` | NIfTI I/O helpers |
