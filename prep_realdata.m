addpath(genpath('~/Documents/MATLAB/NeuroDOT'))
addpath(genpath('../NIRFASTer'))
clear
%%
dataset='CCW1'; % CCW1, CCW2, CW1, IN1, OUT1
load(['NeuroDOT_Data_Sample_',dataset,'.mat']); % data, info, flags
%% Preprocessing: same as NeuroDOT demo
lmdata = logmean(data);                                 % Logmean Light Levels
info = FindGoodMeas(lmdata, info, 0.075);               % Detect Noisy Channels
lmdata = detrend_tts(lmdata);                           % Detrend Data
lmdata = highpass(lmdata, .02, info.system.framerate);  % High Pass Filter (0.02 Hz)
lmdata = lowpass(lmdata, 1, info.system.framerate);     % Low Pass Filter 1 (1.0 Hz)
hem = gethem(lmdata, info);                             % Superficial Signal Regression
[lmdata, ~] = regcorr(lmdata, info, hem);
lmdata = lowpass(lmdata, 0.5, info.system.framerate);   % Low Pass Filter 2 (0.5 Hz)
[lmdata, info] = resample_tts(lmdata, info, 1, 1e-5);   % 1 Hz Resampling (1 Hz)

%%
mesh = load_mesh('Example_Mesh');
% Interpolate data onto a regular grid; feel free to change the resolution,
% but dont' go too high
xgrid = -88:2:88;
ygrid = -118:2:84;
zgrid = -74:2:100;
mesh = gen_intmat(mesh, xgrid, ygrid, zgrid);

fprintf("Calculating Jacobian 850nm\n")
J = jacobiangrid_stnd_FD(mesh);
tmpinfo = load('Pad_28x24Mod_on_mesh.mat','info');
link = [tmpinfo.info.pairs.Src, tmpinfo.info.pairs.Det];
% the original is repeated twice, because of the two wavelengths used
% Using only the first half is therefore sufficient
link0 = [info.pairs.Src(1:672), info.pairs.Det(1:672)];

% There are more channels in the dataset than we have in the model
% We need to find the right subset from the real dataset
idx = zeros(size(link,1), 1);
for i=1:length(idx)
    tmp = find(link0(:,1)==link(i,1) & link0(:,2)==link(i,2));
    if ~isempty(tmp)
        idx(i) = tmp;
    end
end

% select the channels that are present in the model
lmdata750 = lmdata(idx,:);
lmdata850 = lmdata(idx+672,:);
keep750 = info.MEAS.GI(idx);
keep850 = info.MEAS.GI(idx+672);

% Reconstruction, at 850nm
% Ref: https://doi.org/10.1038/nphoton.2014.107 (supplementary)
L = sqrt(0.1 + sum(J.complete.^2));
Ap = J.complete./L;
[~, inv_op850] = tikhonov(Ap(keep850,:), 0.01);
inv_op850 = inv_op850./L';
mua_recon_850 = inv_op850*lmdata850(keep850,:);

% Total number of time steps
dt  = 36;
% The grid is only to the back of the head
% Let's define a subregion to feed into the neural net
x_idx = 14:77;
y_idx = 1:32;
z_idx = 25:64;
% Block average function in NeuroDOT - see the full processing demo
ba_recon_850vec = BlockAverage(mua_recon_850, info.paradigm.synchpts(info.paradigm.Pulse_2), dt);
ba_recon_850 = reshape(ba_recon_850vec, length(ygrid), length(xgrid), length(zgrid), dt);
noisy850 = ba_recon_850(y_idx, x_idx, z_idx, :);
% std850 = zeros(length(dt), 1);
for i=1:dt
    tmp=noisy850(:,:,:,i);
    % std850(i) = std(tmp(:));
    noisy850(:,:,:,i) = tmp/std(tmp(:)); % this is only for the neural net
end
% save('CCW1_850_noisy', 'noisy850', 'std850');

% Same procedure for the other wavelength
fprintf("Calculating Jacobian 750nm\n")
mesh = load_mesh('Example_Mesh750');
mesh = gen_intmat(mesh, xgrid, ygrid, zgrid);
J = jacobiangrid_stnd_FD(mesh);

L = sqrt(0.1 + sum(J.complete.^2));
Ap = J.complete./L;
[~, inv_op750] = tikhonov(Ap(keep750,:), 0.01);
inv_op750 = inv_op750./L';

mua_recon_750 = inv_op750*lmdata750(keep750,:);
ba_recon_750vec = BlockAverage(mua_recon_750, info.paradigm.synchpts(info.paradigm.Pulse_2), dt);
ba_recon_750 = reshape(ba_recon_750vec, length(ygrid), length(xgrid), length(zgrid), dt);
noisy750 = ba_recon_750(y_idx, x_idx, z_idx, :);
% std750 = zeros(length(dt), 1);
for i=1:dt
    tmp=noisy750(:,:,:,i);
    % std750(i) = std(tmp(:));
    noisy750(:,:,:,i) = tmp/std(tmp(:)); % this is only for the neural net
end
% save('CCW1_750_noisy', 'noisy750', 'std750');

if ~exist('E', 'var'),load('E.mat'),end
% Beer-Lambert law; using NeuroDOT functions
% It's better to do BeerLambert on unaveraged data, and THEN calcualte
% block average again
allHb = spectroscopy_img(cat(3, mua_recon_750, mua_recon_850), E);
ba_HbO = BlockAverage(allHb(:,:,1), info.paradigm.synchpts(info.paradigm.Pulse_2), dt);
% ba_HbO=bsxfun(@minus,ba_HbO,ba_HbO(:,1));
HbO = reshape(ba_HbO, length(ygrid), length(xgrid), length(zgrid), dt);
noisy_hbo = HbO(y_idx, x_idx, z_idx, :);
for i=1:dt
    tmp=noisy_hbo(:,:,:,i);
    noisy_hbo(:,:,:,i) = tmp/std(tmp(:)); % this is only for the neural net
end
% save('CCW1_HbO', 'noisy_hbo')

%% Plot results
[~,infoB]=LoadVolumetricData('Segmented_MNI152nl_on_MNI111_nifti',[],'nii');
load('MNI164k_big.mat')
% load dim.mat
% copied from NeuroDOT demo. It's not intuitive how they are calculated
infoA = [];
infoA.center = [-91,-87,-103];
infoA.nVx = 89;
infoA.nVy = 102;
infoA.nVz = 88;
infoA.mmppix = [-2, -2, -2];

% same grid as used in data generation
xgrid = -88:2:88;
ygrid = -118:2:84;
zgrid = -74:2:100;
% this is the subvolume we reconstructed
x_idx = 14:77;
y_idx = 1:32;
z_idx = 25:64;

%% continued
vis1 = zeros(89, 102, 88, 36);
vis2 = zeros(89, 102, 88, 36);

for i=1:36
    tmp = zeros(102, 89, 88);
    tmp(y_idx, x_idx, z_idx) = noisy_hbo(:,:,:,i);
    tmp = permute(tmp, [2,1,3]);
    vis1(:,:,:,i) = tmp;
end

recon1_atlas = affine3d_img(vis1,infoA,infoB,eye(4));

pS.Scale=10;     % Scale wrt/max of data
pS.Th.P=1;            % Threshold to see strong activations
pS.Th.N=-100; 
pS.view='post'; % Posterior view
pS.ctx='std'; % Standard pial cortical view

%% Write the results to a video
% you can also visualize a selected frame, of course
for i=1:36
    PlotInterpSurfMesh(recon1_atlas(:,:,:,i), MNIl,MNIr, infoB, pS);
    pause(0.1);
    frame = getframe(gcf);
    allframes{i} = frame;
    close(gcf)
end

v = VideoWriter('hbo_noisy.avi');
v.FrameRate = 4;
open(v);
for i=1:36
    v.writeVideo(allframes{i});
end
close(v)
