addpath(genpath('../MatlabPackages'))
clear


load("Datasets/FinalModel/test_processed.mat")

A_fn='CroppedWithjoenVtoggle/A_AdultV24x28_onHD_Mesh0_test.mat';   % Sensitivity Matrix

video = true;

if ~exist('A', 'var')       % In case running by hand or re-running script
    A=load([A_fn],'info','A');
    if length(size(A.A))>2  % A data structure [wl X meas X vox]-->[meas X vox]
        [Nwl,Nmeas,Nvox]=size(A.A);
        A.A=reshape(permute(A.A,[2,1,3]),Nwl*Nmeas,Nvox);
    end        
end

[~,infoB]=LoadVolumetricData('Segmented_MNI152nl_on_MNI111_nifti',[],'nii');
load('MNI164k_big.mat')

Params.Scale=1e-3;
Params.Th.P=1e-4;
Params.Th.N=-Params.Th.P;

infoA = A.info.tissue.dim;


% same grid as used in data generation
xgrid = -79:2:79;
ygrid = -111:2:-49;
zgrid = -58:2:68;
% this is the subvolume we reconstructed
x_idx = 1:80;
y_idx = 1:32;
z_idx = 1:64;


%% Noisy
i=10;


vis1 = zeros(80, 32, 64, 34);


recon1_atlas = affine3d_img(HbOvol(:,:,:,i),infoA,infoB,eye(4));


pS.view='post'; % Posterior view
pS.ctx='std'; % Standard pial cortical view
ps.Cmap='jet';
pS.CBar_on = 0;
pS.fig_size = [20, 200, 500, 420];

tp_Eg_atlas=squeeze(recon1_atlas);

PlotInterpSurfMesh(tp_Eg_atlas, MNIl,MNIr, infoB, pS);


%% Smoothed
recon1_atlas = affine3d_img(HbOvol_smoothed(:,:,:,i),infoA,infoB,eye(4));


tp_Eg_atlas=squeeze(recon1_atlas);

PlotInterpSurfMesh(tp_Eg_atlas, MNIl,MNIr, infoB, pS);

%% UNET
load('mask.mat');
recon1_atlas = affine3d_img(HbOvol_unet(:,:,:,i).*mask,infoA,infoB,eye(4));


tp_Eg_atlas=squeeze(recon1_atlas);

PlotInterpSurfMesh(tp_Eg_atlas, MNIl,MNIr, infoB, pS);
%% Ground Truth
recon1_atlas = affine3d_img(ground_truth(:,:,:,i),infoA,infoB,eye(4));


tp_Eg_atlas=squeeze(recon1_atlas);

PlotInterpSurfMesh(tp_Eg_atlas, MNIl,MNIr, infoB, pS);