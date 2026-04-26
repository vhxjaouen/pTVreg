import os
import subprocess
import argparse

# Arguments
parser = argparse.ArgumentParser(description="Batch registration for dataset using ptvreg CLI")
parser.add_argument("dataset_root", help="Root directory of the dataset (contains subject folders)")
parser.add_argument("--ptvreg_path", default="ptvreg", help="Path to ptvreg CLI (default: ptvreg in PATH)")
parser.add_argument("--iterations", nargs='+', default=[1500, 1000, 1000], type=int, help="Iterations per level")
parser.add_argument("--lambda_reg", nargs='+', default=[0.1, 0.15, 0.2], type=float, help="Regularization weights per level")
parser.add_argument("--scale_factor", default=0.5, type=float, help="Scale factor for registration")
parser.add_argument("--dry_run", action="store_true", help="Print commands without running them")
args = parser.parse_args()

root = args.dataset_root
subjects = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

for subj in subjects:
    subj_dir = os.path.join(root, subj)
    fixed = os.path.join(subj_dir, "ct-label", "ct_nobg.nii.gz")
    moving = os.path.join(subj_dir, "features", "mri_combined_sum_noface.nii.gz")
    warp = os.path.join(subj_dir, "features", "ptvreg_warp.nii.gz")
    output = os.path.join(subj_dir, "features", "mri_ptvreg.nii.gz")

    cmd = [
        args.ptvreg_path,
        "--iterations", *(str(i) for i in args.iterations),
        "--lambda_reg", *(str(l) for l in args.lambda_reg),
        "--scale_factor", str(args.scale_factor),
        "-f", fixed,
        "-m", moving,
        "-w", warp,
        "-o", output
    ]

    print("Running:", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)
        print(f"Done {subj}")
    else:
        print(f"Dry run for {subj}")
