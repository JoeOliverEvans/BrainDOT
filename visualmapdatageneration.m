addpath(genpath('../MatlabPackages'))
clear
%% Generate data
solver=get_solver('BiCGStab_GPU');
opt = solver_options;
opt.GPU = 0;
% load the mesh for 850nm
mesh850 = load_mesh('JoeMeshCCW1/CCW1Mesh');
load("mask.mat")



% xgrid = -88:2:88;
% ygrid = -118:2:84;
% zgrid = -74:2:100;
xgrid = -79:2:79;
ygrid = -119:2:-57;
zgrid = -58:2:68;
mesh850 = gen_intmat(mesh850, xgrid, ygrid, zgrid);
mesh750 = mesh850;
J850 = jacobian_stnd_FD(mesh850,0,[],solver,opt);
% Trick 1: find the voxels that are in the gray matter, AND the grid is
% sensitive to
gray=mesh850.region==3 & sum(J850.complete.^2)'>1e-4;
% 750nm mesh - change the optical properties of the 850nm mesh, or load the
% previously saved mesh. Should make no difference
region = mesh750.region;
mesh750.mua(region==2) = 0.0167;
mesh750.mua(region==3) = 0.0180;
mesh750.mua(region==4) = 0.0116;
mesh750.mua(region==5) = 0.0170;

mesh750.mus(region==2) = 1.1908;
mesh750.mus(region==3) = 0.8359;
mesh750.mus(region==4) = 0.94;
mesh750.mus(region==5) = 0.74;
% Don't forget to change kappa
mesh750.kappa = 1./(3*(mesh750.mua + mesh750.mus));
% J750 = jacobian_stnd_FD(mesh750);

load E.mat % File containing extinction coefficients of HbO and Hb
% [E_{750nm, HbO}, E_{750nm, Hb}]
% [E_{850nm, HbO}, E_{850nm, Hb}]

gray_coord=mesh750.nodes(gray,:);
N_gray = sum(gray);
N_nodes = size(mesh750.nodes,1);

%% Important
samples = nnz(mask);
nchannel = size(mesh750.link, 1);
% Random amplitude of dHbO, both positive and negative
% dHbO should only be positive during activation, but since real data is
% centered around its mean, the *processed* data will swing in both sides
all_amplitude = 0.1 * ones(samples);%(rand(samples, 2)*0.1+0.05) .* (((rand(samples, 2)>0.5)-0.5)*2);

all_x = [];
all_y = [];
all_z = [];

size(mask)

for x=1:size(mask, 1)
    for y=1:size(mask, 2)
        for z=1:size(mask, 3)
            if mask(x,y,z) == 1
                all_x = [all_x, xgrid(y)];
                all_y = [all_y, ygrid(x)];
                all_z = [all_z, zgrid(z)];
            end
        end
    end
end


all_dOD = zeros(nchannel, samples);
% assume that dHb is a fraction of dHbO; usually around 3, here we
% randomize between 1~4

all_ratio = rand(samples)*1.5 + 2.5;
togrid = mesh750.vol.mesh2grid;

J750 = jacobiangrid_stnd_FD(mesh750,[],[],[],0,solver,opt);
J850 = jacobiangrid_stnd_FD(mesh850,[],[],[],0,solver,opt);



for rep=1:samples
    fprintf('%d/%d\n', rep, samples);

    % The dHbO vector in source space
    beta = zeros(N_nodes,1);
    tol = 1;
    idx = squeeze(mesh750.nodes(:, 1)>all_x(rep)-tol & mesh750.nodes(:,1)<all_x(rep)+tol & mesh750.nodes(:,2)>all_y(rep)-tol & mesh750.nodes(:,2)<all_y(rep)+tol & mesh750.nodes(:,3)>all_z(rep)-tol & mesh750.nodes(:,3)<all_z(rep)+tol);
    beta(idx) = all_amplitude(rep);

    % for each wavelength, dOD = J * dmua; dmua = E(hbo)*dHbO+E(Hb)*dHb
    % "beer-lambert law"
    % Also, we assuumed dHb=-dHbo/ratio (HbO increase, Hb decrease)
    % Also, we need to convert the brain source vector to grid space
    % ...and we need to concatentate the dOD for the two wavelengths
    all_dOD(:,rep) = [J750.complete(1:672, :)*(togrid*(E(1,1)*beta-E(1,2)*beta/all_ratio(rep))); J850.complete(1:672, :)*(togrid*(E(2,1)*beta-E(2,2)*beta/all_ratio(rep)))];
end
% Add 2% noise to dOD
all_noise = 0.02*rand(samples,1);
amp = max(abs(all_dOD));
all_dOD_noisy = all_dOD + amp.*all_noise'.*randn(size(all_dOD));

% save('tmp_spec4','-v7.3')

%% Reconstruction
fprintf("Reconstruction...\n")


load E.mat
% Apply beer-lambert law to get the "spectral" jacobian
J_spec = [[J750.complete(1:672,:)*E(1,1), J750.complete(1:672,:)*E(1,2)]; [J850.complete(1:672,:)*E(2,1), J850.complete(1:672,:)*E(2,2)]];
% This step uses a LOT of memory and can crash Matlab
% might be a good idea to save the results before running these lines
L = sqrt(0.1 + sum(J_spec.^2));
Ap = J_spec./L;
[~, inv_op] = tikhonov(Ap, 0.01);
inv_op = inv_op./L';

% load invop

all_recon = inv_op * all_dOD_noisy;
all_recon = all_recon(1:end/2,:);
all_recon = reshape(all_recon, length(ygrid), length(xgrid), length(zgrid), samples);
% carve out only the posterior part of the head, same as in real data
x_idx = 1:80;
y_idx = 1:32;
z_idx = 1:64;

noisy_images = zeros(length(y_idx), length(x_idx), length(z_idx), samples);
clean_images = zeros(length(y_idx), length(x_idx), length(z_idx), samples);
for i=1:samples
    tmp = all_recon(y_idx,x_idx,z_idx,i);
    noisy_images(:,:,:,i) = flip(tmp./std(tmp(:)), 2);
    if(any(isnan(tmp(:))))
        fprintf('Nan detected %d\n', i);
    end
end



fprintf("Saving...\n")
save('Datasets/images_CCW1Mesh_visualmap.mat','clean_images','noisy_images','mask', '-v7.3');
save('Datasets/Data_CCW1Mesh_visualmap.mat', 'all_recon', 'all_x', 'all_y', 'all_z', 'all_amplitude', 'all_dOD', 'all_dOD_noisy', 'all_noise','all_ratio', 'samples', '-v7.3');

