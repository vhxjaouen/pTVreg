import argparse
import numpy as np
import nibabel as nib
import optuna
import torch
import gc
from scipy import ndimage
from pTVreg.registration import PTVRegistration

def compute_nmi(img1, img2, bins=64):
    """
    Compute Normalized Mutual Information (NMI)
    Metric de choix pour le multimodal (CT / MRI). 
    Maximized quand les images partagent le plus d'informations structurelles.
    """
    f_flat = img1.flatten()
    w_flat = img2.flatten()
    
    # Calculate joint histogram
    hist_2d, _, _ = np.histogram2d(f_flat, w_flat, bins=bins, range=[[0, 1], [0, 1]])
    
    # Convert to probabilities
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    eps = 1e-8
    hx = -np.sum(px * np.log2(px + eps))
    hy = -np.sum(py * np.log2(py + eps))
    hxy = -np.sum(pxy * np.log2(pxy + eps))
    
    nmi = (hx + hy) / (hxy + eps)
    return nmi

def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Search for pTVreg")
    parser.add_argument('-f', '--fixed', required=True, help="Fixed NIfTI image")
    parser.add_argument('-m', '--moving', required=True, help="Moving NIfTI image")
    parser.add_argument('-n', '--n-trials', type=int, default=30, help="Number of search iterations")
    parser.add_argument('--scale', type=float, default=0.5, help="Downscale images to search 10x faster")
    args = parser.parse_args()

    print(f"Loading and strictly normalizing images (scale={args.scale})...")
    fixed_img = nib.load(args.fixed).get_fdata().astype(np.float32)
    moving_img = nib.load(args.moving).get_fdata().astype(np.float32)
    
    if args.scale != 1.0:
        fixed_img = ndimage.zoom(fixed_img, args.scale, order=1)
        moving_img = ndimage.zoom(moving_img, args.scale, order=1)
        
    fixed_norm = (fixed_img - np.min(fixed_img)) / (np.ptp(fixed_img) + 1e-8)
    moving_norm = (moving_img - np.min(moving_img)) / (np.ptp(moving_img) + 1e-8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def objective(trial):
        # 1. AI decides the next best parameters based on previous trials (Bayesian Optimization)
        metric = trial.suggest_categorical('metric', ['lcc', 'ngf', 'emse'])
        lambda_reg = trial.suggest_float('lambda_reg', 0.05, 0.4, log=True)
        grid_spacing = trial.suggest_categorical('grid_spacing', [3, 4, 6])
        n_levels = trial.suggest_int('n_levels', 2, 4)
        
        # Conditional logic based on metric type
        if metric == 'lcc':
            metric_param = trial.suggest_float('metric_param_lcc', 1.5, 4.0)
        elif metric == 'ngf':
            metric_param = trial.suggest_float('metric_param_ngf', 0.01, 0.1)
        else:
            metric_param = 0.0 # Emse doesn't use tweaking parameters

        print(f"\n[Trial {trial.number}] Trying metric={metric}, reg={lambda_reg:.3f}, grid={grid_spacing}, levels={n_levels}")

        try:
            reg = PTVRegistration(fixed_norm, moving_norm, 
                                  grid_spacing=grid_spacing,
                                  lambda_reg=lambda_reg,
                                  n_levels=n_levels,
                                  metric=metric,
                                  metric_param=metric_param,
                                  device=device)
            
            # Less iterations to quickly prototype parameter directions
            iters = [40] * n_levels 
            warped, _ = reg.optimize(iterations=iters)
            
            if hasattr(warped, 'detach'):
                warped_np = warped.detach().cpu().numpy().squeeze()
            else:
                warped_np = warped
            
            # Flush VRAM for the next trial
            del reg, warped, _
            torch.cuda.empty_cache()
            gc.collect()

            # 2. Our "Juge de paix": Normalized Mutual Information
            score = compute_nmi(fixed_norm, warped_np)
            return score

        except Exception as e:
            # If the specific parameters cause a crash (OOM, divergent math), abort this specific trial smoothly
            print(f"Trial failed: {e}")
            raise optuna.exceptions.TrialPruned()

    print("\nStarting Optuna Study. Objective: MAXIMIZE NMI.")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)

    print("\n==============================")
    print(f"BEST NMI SCORE: {study.best_value:.4f}")
    print("BEST PARAMETERS FOUND:")
    for k, v in study.best_params.items():
        print(f"  --{k.split('_')[0] if 'metric_param' in k else k} : {v}")
    print("==============================\n")

if __name__ == '__main__':
    main()
