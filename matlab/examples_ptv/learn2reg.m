clearvars
close all

addpath(genpath('../mutils/My/'));
addpath(genpath('../ptv'));
addpath('/home/ziqiao/pCloudDrive/CODE/MATLAB/_fileExchange/toolsForAnalyzeAndNifti')

opts = [];
fast_lcc = 1;

opts.grid_spacing = [4, 4, 4];  
opts.cp_refinements = 0;
opts.loc_cc_approximate = fast_lcc;

opts.display = 'off';
opts.k_down = 0.7;
opts.interp_type = 0;
opts.nlvl = []; 
opts.metric = 'loc_cc_fftn_gpu'; % default
% opts.metric = 'ngf';
opts.metric_param = [1,1,1] * 2.1;
opts.scale_metric_param = true;
opts.isoTV = 0.11;
opts.csqrt = 5e-3;
opts.spline_order = 1;
opts.border_mask = 5;
opts.max_iters =  80;
opts.check_gradients = 100*0;



fname_mov = '/data/PUBLIC/LEARN2REG/ThoraxCBCT/imagesTr/ThoraxCBCT_0000_0001.nii.gz';
fname_fix = '/data/PUBLIC/LEARN2REG/ThoraxCBCT/imagesTr/ThoraxCBCT_0000_0000.nii.gz';

fname_mov = '/home/ziqiao/DATA/PUBLIC/ThoraxCBCT/imagesTr/ThoraxCBCT_0000_0002.nii.gz';
fname_fix = '/home/ziqiao/DATA/PUBLIC/ThoraxCBCT/imagesTr/ThoraxCBCT_0000_0000.nii.gz';

volmov_nii = load_untouch_nii(fname_mov);
volfix_nii = load_untouch_nii(fname_fix);
opts.pix_resolution = volfix_nii.hdr.dime.pixdim(1:3);

volmov = volmov_nii.img;
volfix = volfix_nii.img;

% volmov = img_thr(volmov, -300, 300, 1);
% volfix = img_thr(volfix, -300, 300, 1);


t0 = tic();
[voldef, Tptv, Kptv] = ptv_register(volmov, volfix, opts);
t1=toc(t0);

figure, imagesc(volfix(:,:,100))

% ---------- COMPUTE VFC


