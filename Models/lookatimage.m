addpath(genpath('../../MatlabPackages'))
clear


load("BiggerDatasetDeeper\CCW1\CCW1_processed.mat")

A_fn='A_AdultV24x28_onHD_Mesh0_test.mat';   % Sensitivity Matrix

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

l = 16

%% continued
vis1 = zeros(80, 32, 64, 34);
vis2 = zeros(80, 32, 64, 34);

for i=1:36
    tmp = zeros(32, 80, 64);
    tmp(y_idx, x_idx, z_idx) = smooth_images(:,:,:,i);
    tmp = permute(tmp, [2,1,3]);
    vis1(:,:,:,i) = tmp;
end

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

if video == true
    for i=1:36
        PlotInterpSurfMesh(recon1_atlas(:,:,:,i), MNIl,MNIr, infoB, pS);
        pause(0.1);
        frame = getframe(gcf);
        allframes{i} = frame;
        close(gcf)
    end
    
    v = VideoWriter('smooth_images.avi');
    v.FrameRate = 4;
    open(v);
    for i=1:36
        v.writeVideo(allframes{i});
    end
    close(v)
end

%% continued
vis1 = zeros(80, 32, 64, 34);
vis2 = zeros(80, 32, 64, 34);

for i=1:36
    tmp = zeros(32, 80, 64);
    tmp(y_idx, x_idx, z_idx) = noisy_images(:,:,:,i);
    tmp = permute(tmp, [2,1,3]);
    vis1(:,:,:,i) = tmp;
end

recon1_atlas = affine3d_img(vis1,infoA,infoB,eye(4));



tp_Eg_atlas=squeeze(recon1_atlas(:,:,:,l));

PlotInterpSurfMesh(tp_Eg_atlas, MNIl,MNIr, infoB, pS);

if video == true
    for i=1:36
        PlotInterpSurfMesh(recon1_atlas(:,:,:,i), MNIl,MNIr, infoB, pS);
        pause(0.1);
        frame = getframe(gcf);
        allframes{i} = frame;
        close(gcf)
    end
    
    v = VideoWriter('noisy_images.avi');
    v.FrameRate = 4;
    open(v);
    for i=1:36
        v.writeVideo(allframes{i});
    end
    close(v)
end
%% continued
vis1 = zeros(80, 32, 64, 34);
vis2 = zeros(80, 32, 64, 34);

for i=1:36
    tmp = zeros(32, 80, 64);
    tmp(y_idx, x_idx, z_idx) = recon2(:,:,:,i);
    tmp = permute(tmp, [2,1,3]);
    vis1(:,:,:,i) = tmp;
end

recon1_atlas = affine3d_img(vis1,infoA,infoB,eye(4));



tp_Eg_atlas=squeeze(recon1_atlas(:,:,:,l));

PlotInterpSurfMesh(tp_Eg_atlas, MNIl,MNIr, infoB, pS);



if video == true
    for i=1:36
        PlotInterpSurfMesh(recon1_atlas(:,:,:,i), MNIl,MNIr, infoB, pS);
        pause(0.1);
        frame = getframe(gcf);
        allframes{i} = frame;
        close(gcf)
    end
    
    v = VideoWriter('recon2.avi');
    v.FrameRate = 4;
    open(v);
    for i=1:36
        v.writeVideo(allframes{i});
    end
    close(v)
end