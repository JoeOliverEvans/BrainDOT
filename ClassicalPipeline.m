addpath(genpath('../MatlabPackages'))


dataset = 'CCW1';
load(['NeuroDOT_Data_Sample_',dataset,'.mat']); % data, info, flags

% Set parameters for A and block length for quick processing examples

A_fn='A_AdultV24x28_onHD_Mesh0_test.mat';   % Sensitivity Matrix
dt=36;                      % Block length
tp=16;                      % Example (block averaged) time point

%% View data before filtering
omega_lp1 = 1;

if info.system.framerate/2 < omega_lp1 
    % Adjust Lowpass filter cutoff
    % frequency for systems with lower framerates
    omega_lp1 = (info.system.framerate/2)*0.90;
end

%% PRE-PREOCESSING PIPELINE
% Note: the first 3 lines are repeated from above but with changed variable names
lmdata = logmean(data);                                                   % Logmean Light Levels
info = FindGoodMeas(lmdata, info, 0.075);                                 % Detect Noisy Channels
lmdata = detrend_tts(lmdata);                                             % Detrend Data
lmdata = highpass(lmdata, .02, info.system.framerate);                    % High Pass Filter (0.02 Hz)
lmdata = lowpass(lmdata, omega_lp1, info.system.framerate);                       % Low Pass Filter 1 (1.0 Hz)
hem = gethem(lmdata, info);                                               % Superficial Signal Regression
[lmdata, ~] = regcorr(lmdata, info, hem);
lmdata = lowpass(lmdata, 0.5, info.system.framerate);                     % Low Pass Filter 2 (0.5 Hz)
[lmdata, info] = resample_tts(lmdata, info, 1, 1e-5);                     % 1 Hz Resampling (1 Hz)
[info.GVTD, info.DQ_metrics.med_GVTD] = CalcGVTD(lmdata(info.MEAS.GI & info.pairs.r2d<20,:));         % Calculate GVTD

%% Block Averaging the measurement data and view
badata = BlockAverage(lmdata, info.paradigm.synchpts(info.paradigm.Pulse_2), dt);
badata=bsxfun(@minus,badata,mean(badata(:,1:4),2));

%% RECONSTRUCTION PIPELINE
if ~exist('A', 'var')       % In case running by hand or re-running script
    A=load([A_fn],'info','A');
    if length(size(A.A))>2  % A data structure [wl X meas X vox]-->[meas X vox]
        [Nwl,Nmeas,Nvox]=size(A.A);
        A.A=reshape(permute(A.A,[2,1,3]),Nwl*Nmeas,Nvox);
    end        
end
Nvox=size(A.A,2);
Nt=size(lmdata,2);
cortex_mu_a=zeros(Nvox,Nt,2);

for j = 1:2
    keep = (info.pairs.WL == j) & (info.pairs.r2d <= 40) & info.MEAS.GI; % This handles the two wavelengths separately
    disp('> Inverting A')                
    iA = Tikhonov_invert_Amat(A.A(keep, :), 0.01, 0.1); % Invert A-Matrix
    disp('> Smoothing iA')
    iA = smooth_Amat(iA, A.info.tissue.dim, 3);         % Smooth Inverted A-Matrix      
    cortex_mu_a(:, :, j) = reconstruct_img(lmdata(keep, :), iA);% Reconstruct Image Volume
end

%% Spectroscopy
if ~exist('E', 'var')
    load('E.mat')
end
cortex_Hb = spectroscopy_img(cortex_mu_a, E);
cortex_HbO = cortex_Hb(:, :, 1);
cortex_HbR = cortex_Hb(:, :, 2);
cortex_HbT = cortex_HbO + cortex_HbR;

%% Select Volumetric visualizations of block averaged data
if ~exist('MNI', 'var')
[MNI,infoB]=LoadVolumetricData('Segmented_MNI152nl_on_MNI111_nifti',[],'nii'); % Load MRI (same data set as in A matrix dim)
end
MNI_dim = affine3d_img(MNI,infoB,A.info.tissue.dim,eye(4),'nearest'); % Transform to DOT volume space

% Block Average Data
badata_HbO = BlockAverage(cortex_HbO, info.paradigm.synchpts(info.paradigm.Pulse_2), dt);
badata_HbO=bsxfun(@minus,badata_HbO,badata_HbO(:,1));
badata_HbOvol = Good_Vox2vol(badata_HbO,A.info.tissue.dim);
tp_Eg=squeeze(badata_HbOvol(:,:,:,tp));

% Explore PlotSlices - The basics (Slide 22 in ppt)
PlotSlices(MNI_dim)                             % Anatomy only
PlotSlices(MNI_dim,A.info.tissue.dim)           % Anatomy + volumetric data

% Visualize the data (Slide 23 in ppt)
PlotSlices(tp_Eg,A.info.tissue.dim);            % Data by itself
PlotSlices(MNI_dim,A.info.tissue.dim,[],tp_Eg); % Data with anatomical underlay
% Set parameters to visualize more specific aspects of data
Params.Scale=0.8*max(abs(tp_Eg(:)));     % Scale wrt/max of data
Params.Th.P=0.4*Params.Scale;            % Threshold to see strong activations
Params.Th.N=-0.010;                % Thresholds go both ways
Params.Cmap='jet';
PlotSlices(MNI_dim,A.info.tissue.dim,Params,tp_Eg);

% Explore the block-averaged data a bit more interactively (slide 24 in ppt)
Params.Scale=0.8*max(abs(badata_HbOvol(:)));
Params.Th.P=0;
Params.Th.N=-Params.Th.P;
PlotSlicesTimeTrace(MNI_dim,A.info.tissue.dim,Params,badata_HbOvol,info)

save('TestingOutputs/badata_HbOvol_file', "badata_HbOvol", '-v7.3')


% Explore the not-block-averaged data a bit more interactively (slide 24 in ppt)
HbOvol = Good_Vox2vol(cortex_HbO,A.info.tissue.dim);
Params.Scale=1e-3;
Params.Th.P=1e-4;
Params.Th.N=-Params.Th.P;
PlotSlicesTimeTrace(MNI_dim,A.info.tissue.dim,Params,HbOvol,info)


%% Select Surface visualizations
if ~exist('MNIl', 'var'),load(['MNI164k_big.mat']);end
HbO_atlas = affine3d_img(badata_HbOvol,A.info.tissue.dim,infoB,eye(4));
tp_Eg_atlas=squeeze(HbO_atlas(:,:,:,tp));
pS=Params;
pS.view='post';

pS.ctx='std'; % Standard pial cortical view
PlotInterpSurfMesh(tp_Eg_atlas, MNIl,MNIr, infoB, pS);

pS.ctx='inf'; % Inflated pial cortical view
PlotInterpSurfMesh(tp_Eg_atlas, MNIl,MNIr, infoB, pS);

pS.ctx='vinf';% Very Inflated pial cortical view
PlotInterpSurfMesh(tp_Eg_atlas, MNIl,MNIr, infoB, pS);

