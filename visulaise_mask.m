addpath(genpath('../../MatlabPackages'))
clear


load("mask.mat")
%load('../Datasets/SpinningNoFluctuate/images_rotating_nofluctuate.mat')
A_fn='A_AdultV24x28_onHD_Mesh0_test.mat';   % Sensitivity Matrix

video = false;

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

l = 1
%% continued
vis1 = zeros(80, 32, 64, 34);
vis2 = zeros(80, 32, 64, 34);

i=1
vis1(:,:,:,i) = permute(mask, [2,1,3]);

recon1_atlas = affine3d_img(vis1,infoA,infoB,eye(4));

pS.Scale=10;     % Scale wrt/max of data
pS.Th.P=1;            % Threshold to see strong activations
pS.Th.N=-1; 
pS.view='post'; % Posterior view
pS.ctx='std'; % Standard pial cortical view
ps.Cmap='jet';
pS.CBar_on = 1;


tp_Eg_atlas=squeeze(recon1_atlas(:,:,:,l));

PlotInterpSurfMesh(tp_Eg_atlas, MNIl,MNIr, infoB, pS);