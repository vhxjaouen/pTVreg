clearvars; close all;

% Define the main directory
base_dir = "/amadeo/DATA/PUBLIC/L2R2026/PSMAReg_dataset/imagesTr";

% Get a list of all subdirectories
dir_info = dir(base_dir);
subdirs = {dir_info([dir_info(:).isdir]).name}; % Only keep directories
subdirs = subdirs(~ismember(subdirs, {'.', '..'})); % Remove '.' and '..'

% Add necessary paths
addpath('/home/jet/pCloudDrive/GITHUB/pTVreg/ptv/')
addpath(genpath('/home/jet/pCloudDrive/GITHUB/pTVreg/mutils/My/'));

% Loop over each patient folder+6
    
% Construct file names based on the pattern
FIXED_fname = fullfile(base_dir, "PSMARegPSMA_0006_0000_00.nii.gz");
MOVING_fname = fullfile(base_dir,"PSMARegPSMA_0006_0000_02.nii.gz");
pTV_fname = fullfile(base_dir, "PSMARegPSMA_0006_0000_02_pTVreg_matlab.nii.gz");

% Check if both necessary files exist before processing
if exist(FIXED_fname, 'file') && exist(MOVING_fname, 'file')
    
    % Read images
    FIXED = niftiread(FIXED_fname);
    MOVING = niftiread(MOVING_fname);
    
    % Get pixel spacing
    FIXEDINFO = niftiinfo(FIXED_fname);
    spc = FIXEDINFO.PixelDimensions;
    
    % Configure registration options
    opts = [];
    opts.loc_cc_approximate = 1;
    opts.grid_spacing = [4, 4, 3];
    opts.cp_refinements = 0;
    opts.display = 'off';
    opts.k_down = 0.7;
    opts.interp_type = 0;
    opts.metric = 'loc_cc_fftn_gpu';
    opts.metric_param = [1,1,1] * 2.1;
    opts.scale_metric_param = true;
    opts.isoTV = 0.11;
    opts.csqrt = 5e-3;
    opts.spline_order = 1;
    opts.border_mask = 5;
    opts.max_iters = 80;
    opts.check_gradients = 100*0;
    opts.pix_resolution = spc;

    % Perform registration
    [voldef, Tmin_out, Kmin_out, varargout] = ptv_register(MOVING, FIXED, opts);

    % Ensure voldef is in the correct datatype and orientation
    voldef = cast(voldef, FIXEDINFO.Datatype);
    
    % Save the registered volume
    niftiwrite(voldef, pTV_fname, FIXEDINFO, 'Compressed', true);
    fprintf("Saved registered volume as %s\n", pTV_fname);
else
    fprintf("Skipping %s: missing NATIVE or VENOUS file.\n", patient_id);
end

fprintf("Processing complete.\n");
