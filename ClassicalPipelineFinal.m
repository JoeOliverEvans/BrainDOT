addpath(genpath('../MatlabPackages'))
clear

dataset = 'CW1';
load(['NeuroDOT_Data_Sample_',dataset,'.mat']); % data, info, flags

% Set parameters for A and block length for quick processing examples

A_fn='CroppedWithjoenVtoggle/A_AdultV24x28_onHD_Mesh0_test.mat';   % Sensitivity Matrix
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
cortex_mu_a_smoothed=zeros(Nvox,Nt,2);


for j = 1:2
    keep = (info.pairs.WL == j) & (info.pairs.r2d <= 40) & info.MEAS.GI; % This handles the two wavelengths separately
    disp('> Inverting A')                
    iA = Tikhonov_invert_Amat(A.A(keep, :), 0.01, 0.1); % Invert A-Matrix
    disp('> Smoothing iA')
    iA_smoothed = smooth_Amat(iA, A.info.tissue.dim, 3);         % Smooth Inverted A-Matrix
    cortex_mu_a(:, :, j) = reconstruct_img(lmdata(keep, :), iA);% Reconstruct Image Volume
    cortex_mu_a_smoothed(:, :, j) = reconstruct_img(lmdata(keep, :), iA_smoothed);% Reconstruct Image Volume
end


%% Spectroscopy
if ~exist('E', 'var')
    load('E.mat')
end
cortex_Hb = spectroscopy_img(cortex_mu_a, E);
cortex_Hb_smoothed = spectroscopy_img(cortex_mu_a_smoothed, E);

cortex_HbO = cortex_Hb(:, :, 1);
cortex_HbO_smoothed = cortex_Hb_smoothed(:, :, 1);

%%
% Block Average Data
badata_HbO = BlockAverage(cortex_HbO, info.paradigm.synchpts(info.paradigm.Pulse_2), dt);
%badata_HbO=bsxfun(@minus,badata_HbO,badata_HbO(:,1));
badata_HbOvol = Good_Vox2vol(badata_HbO,A.info.tissue.dim);

badata_HbO_smoothed = BlockAverage(cortex_HbO_smoothed, info.paradigm.synchpts(info.paradigm.Pulse_2), dt);
%badata_HbO_smoothed=bsxfun(@minus,badata_HbO_smoothed,badata_HbO_smoothed(:,1));
badata_HbOvol_smoothed = Good_Vox2vol(badata_HbO_smoothed,A.info.tissue.dim);

%Follow previous variable names
HbOvol = badata_HbOvol;
HbOvol_smoothed = badata_HbOvol_smoothed;

save('CW1Neurodot.mat', 'HbOvol', 'HbOvol_smoothed', '-v7.3')

