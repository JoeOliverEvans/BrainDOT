addpath(genpath('../MatlabPackages'))
clear

dataset = 'Fluctuating';
load(['Datasets\Fluctuating\data_rotating_fluctuate.mat']); % data, info, flags

% Set parameters for A and block length for quick processing examples

A_fn='A_AdultV24x28_onHD_Mesh0_test.mat';   % Sensitivity Matrix
dt=36;                      % Block length
tp=16;                      % Example (block averaged) time point

x_idx = 1:80;
y_idx = 1:32;
z_idx = 1:64;


%% RECONSTRUCTION PIPELINE
if ~exist('A', 'var')       % In case running by hand or re-running script
    A=load([A_fn],'info','A');
    if length(size(A.A))>2  % A data structure [wl X meas X vox]-->[meas X vox]
        [Nwl,Nmeas,Nvox]=size(A.A);
        A.A=reshape(permute(A.A,[2,1,3]),Nwl*Nmeas,Nvox);
    end        
end

Nvox=size(A.A,2);
Nt=size(all_dOD_noisy,2);
cortex_mu_a=zeros(Nvox,Nt,2);

for j = 1:2
    keep = (A.info.pairs.WL == j); % This handles the two wavelengths separately
    disp('> Inverting A')                
    iA = Tikhonov_invert_Amat(A.A(keep, :), 0.01, 0.1); % Invert A-Matrix
    disp('> Smoothing iA')
    iA = smooth_Amat(iA, A.info.tissue.dim, 3);         % Smooth Inverted A-Matrix      
    cortex_mu_a(:, :, j) = reconstruct_img(all_dOD_noisy(keep, :), iA);% Reconstruct Image Volume
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
badata_HbO = cortex_HbO;
% badata_HbO=bsxfun(@minus,badata_HbO,badata_HbO(:,1));




badata_HbOvol = Good_Vox2vol(badata_HbO,A.info.tissue.dim);
tp_Eg=squeeze(badata_HbOvol(:,:,:,tp));

%Save data for comparison
badata_HbOvolsave = permute(badata_HbOvol, [2,1,3,4]);
s=size(badata_HbOvolsave);
noisy_images = zeros(s(1), s(2), s(3), s(4));
for i=1:s(4)
    tmp = badata_HbOvolsave(y_idx,x_idx,z_idx,i);
    noisy_images(:,:,:,i) = -tmp./std(tmp(:));
end
smooth_images = noisy_images;
save(['Datasets\', dataset, '\NeurodotSmoothImages'], 'smooth_images', '-v7.3')


%% RECONSTRUCTION PIPELINE
if ~exist('A', 'var')       % In case running by hand or re-running script
    A=load([A_fn],'info','A');
    if length(size(A.A))>2  % A data structure [wl X meas X vox]-->[meas X vox]
        [Nwl,Nmeas,Nvox]=size(A.A);
        A.A=reshape(permute(A.A,[2,1,3]),Nwl*Nmeas,Nvox);
    end        
end
Nvox=size(A.A,2);
Nt=size(all_dOD_noisy,2);
cortex_mu_a=zeros(Nvox,Nt,2);

for j = 1:2
    keep = (A.info.pairs.WL == j); % This handles the two wavelengths separately
    disp('> Inverting A')                
    iA = Tikhonov_invert_Amat(A.A(keep, :), 0.01, 0.1); % Invert A-Matrix
    % disp('> Smoothing iA')
    % iA = smooth_Amat(iA, A.info.tissue.dim, 3);         % Smooth Inverted A-Matrix      
    cortex_mu_a(:, :, j) = reconstruct_img(all_dOD_noisy(keep, :), iA);% Reconstruct Image Volume
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
badata_HbO = cortex_HbO;
% badata_HbO=bsxfun(@minus,badata_HbO,badata_HbO(:,1));




badata_HbOvol = Good_Vox2vol(badata_HbO,A.info.tissue.dim);
tp_Eg=squeeze(badata_HbOvol(:,:,:,tp));

%Save data for comparison
badata_HbOvolsave = permute(badata_HbOvol, [2,1,3,4]);
s=size(badata_HbOvolsave);
noisy_images = zeros(s(1), s(2), s(3), s(4));
for i=1:s(4)
    tmp = badata_HbOvolsave(y_idx,x_idx,z_idx,i);
    noisy_images(:,:,:,i) = -tmp./std(tmp(:));
end
save(['Datasets\', dataset, '\NeurodotImages'], 'noisy_images', '-v7.3')


