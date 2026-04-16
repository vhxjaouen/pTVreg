import re
import sys

def patch_file(filepath, is_warp=False):
    with open(filepath, 'r') as f:
        text = f.read()

    # Add sys import if not present
    if 'import sys' not in text:
        text = text.replace('import argparse\n', 'import argparse\nimport sys\n')

    # Add formatter_class
    if is_warp:
        desc = "description='pTVwarp: Apply warp field to moving image, resampling to reference space.'"
        new_desc = "description='pTVwarp: Apply warp field to moving image, resampling to reference space.', formatter_class=argparse.ArgumentDefaultsHelpFormatter"
    else:
        desc = "description='pTVreg: Python Image Registration'"
        new_desc = "description='pTVreg: Python Image Registration', formatter_class=argparse.ArgumentDefaultsHelpFormatter"
        
    text = text.replace(desc, new_desc)

    # Add sys.argv check
    parse_call = "args = parser.parse_args()"
    new_parse_call = """if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()"""
    text = text.replace(parse_call, new_parse_call)

    if not is_warp:
        # Update help strings with ranges
        text = text.replace("help='Number of pyramid levels'", "help='Number of pyramid levels (indicative range: 3-5)'")
        text = text.replace("help='Grid spacing at finest level (default: 8)'", "help='Grid spacing at finest level, in voxels (indicative: 4, 8, 16)'")
        text = text.replace("help='Regularization weight base, or list per scale (default: 0.15)'", "help='Regularization weight base, or list per scale (indicative range: 0.05 - 0.5)'")
        text = text.replace("help='Iterations per level. Can be a single scalar or a list (coarsest to finest). (default: 160 120 80)'", "help='Iterations per level. Can be a single scalar or a list (coarsest to finest). (indicative: 50-300 per level)'")
        text = text.replace("help='Metric parameter (e.g. LCC sigma, default: 2.1)'", "help='Metric parameter (e.g. LCC sigma). (indicative range: 1.5 - 4.0)'")
        text = text.replace("help='Downsample scale factor (e.g. 0.5 for 50%% size) to save memory. (default: 1.0)'", "help='Downsample scale factor (e.g. 0.5 for 50%% size) to save memory and speed up processing.'")
        
        # Remove manual (default: ...) from help strings as ArgumentDefaultsHelpFormatter handles it
        text = text.replace(" (default: ./moving_warped.nii.gz)", "")
        text = text.replace(" (default: auto-detect)", "")

    with open(filepath, 'w') as f:
        f.write(text)

patch_file('pTVreg/cli.py', False)
patch_file('pTVreg/cli_warp.py', True)
