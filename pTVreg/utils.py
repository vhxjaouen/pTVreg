import torch
import numpy as np
import nibabel as nib
import scipy.io as sio

def to_torch(x, device='cpu', dtype=torch.float32):
    """Convert numpy array to torch tensor with (N, C, ...) shape."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    
    x = np.array(x)
    # Assume input x is (D, H, W) or (H, W) -> add batch and channel dims
    # If already has batch/channel, handle accordingly.
    # For this simple port, we assume input is single volume (D, H, W)
    if x.ndim == 3:
        x = x[np.newaxis, np.newaxis, ...]
    elif x.ndim == 2:
        x = x[np.newaxis, np.newaxis, ...]
        
    return torch.from_numpy(x).to(device=device, dtype=dtype)

def to_numpy(x):
    """Convert torch tensor to numpy array, removing batch/channel dims if 1."""
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy().squeeze()

def load_data(path):
    """Load data from .mat, .nii, or .nii.gz file."""
    if path.endswith('.mat'):
        mat = sio.loadmat(path)
        # Heuristic to find the image variable in mat file
        for k, v in mat.items():
            if isinstance(v, np.ndarray) and v.ndim >= 2 and not k.startswith('__'):
                return v
        raise ValueError(f"No suitable array found in {path}")
    elif path.endswith(('.nii', '.nii.gz')):
        img = nib.load(path)
        return img.get_fdata()
    else:
        raise ValueError(f"Unsupported file format: {path}")

def save_nifti(data, path, affine=np.eye(4)):
    """Save data as NIfTI file."""
    data = to_numpy(data)
    if data.dtype == np.float16:
        data = data.astype(np.float32)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)
