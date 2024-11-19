addpath(genpath('../MatlabPackages'))
clear
%% Generate data
solver=get_solver('BiCGStab_GPU');
opt = solver_options;
opt.GPU = 1;
% load the mesh for 850nm
mesh850 = load_mesh('JoeMeshCCW1/CCW1Mesh');


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
samples = 3000;
nchannel = size(mesh750.link, 1);
% Random amplitude of dHbO, both positive and negative
% dHbO should only be positive during activation, but since real data is
% centered around its mean, the *processed* data will swing in both sides
all_amplitude = (rand(samples, 2)*0.1+0.05) .* (((rand(samples, 2)>0.5)-0.5)*2);
% These are the centers of the activation spot
all_x = zeros(2, samples);
all_y = zeros(2, samples);
all_z = zeros(2, samples);
% all_r = rand(samples,2)*10 + 10; % used in spec3
% randomize the radius of the activation spot
all_r = rand(samples,2)*10 + 5; % used in spec4
all_beta = zeros(N_nodes, samples);
% proportion baseline mua and mus fluctuation
all_fluctuate1 = 0.2*rand(10, samples) - 0.1;
all_fluctuate2 = 0.2*rand(10, samples) - 0.1;
all_dOD = zeros(nchannel, samples);
% assume that dHb is a fraction of dHbO; usually around 3, here we
% randomize between 1~4
all_ratio = rand(samples)*1.5 + 2.5;
togrid = mesh750.vol.mesh2grid;
for rep=1:samples
    fprintf('%d/%d\n', rep, samples);
    while 1
        % Trick: assume activation only happens in the gray matter and
        % where the grid is sensitive to
        % Simulate exactly two activations, and they are at least 40mm away
        % from each other
        centers = gray_coord(randperm(N_gray, 2), :);
        if norm(centers(1,:) - centers(2,:))>40
            break
        end
    end
    all_x(:,rep) = centers(:,1);
    all_y(:,rep) = centers(:,2);
    all_z(:,rep) = centers(:,3);
    % The dHbO vector in source space
    beta = zeros(N_nodes,1);
    for i=1:2
        idx = vecnorm(mesh750.nodes-centers(i,:),2,2)<all_r(rep) & gray;
        beta(idx) = all_amplitude(rep,i);
    end
    all_beta(:,rep) = beta;
    % Calculate the Jacobians, after fluctuating the baseline optical props
    mesh750_2 = mesh750;
    mesh750_2.mua(region==1) = mesh750.mua(region==1)*(1+all_fluctuate1(1,rep));
    mesh750_2.mua(region==2) = mesh750.mua(region==2)*(1+all_fluctuate1(2,rep));
    mesh750_2.mua(region==3) = mesh750.mua(region==3)*(1+all_fluctuate1(3,rep));
    mesh750_2.mua(region==4) = mesh750.mua(region==4)*(1+all_fluctuate1(4,rep));
    mesh750_2.mua(region==5) = mesh750.mua(region==5)*(1+all_fluctuate1(5,rep));
    mesh750_2.mus(region==1) = mesh750.mus(region==1)*(1+all_fluctuate1(6,rep));
    mesh750_2.mus(region==2) = mesh750.mus(region==2)*(1+all_fluctuate1(7,rep));
    mesh750_2.mus(region==3) = mesh750.mus(region==3)*(1+all_fluctuate1(8,rep));
    mesh750_2.mus(region==4) = mesh750.mus(region==4)*(1+all_fluctuate1(9,rep));
    mesh750_2.mus(region==5) = mesh750.mus(region==5)*(1+all_fluctuate1(10,rep));
    mesh750_2.kappa = 1./(3*(mesh750_2.mua + mesh750_2.mus));
    J750 = jacobiangrid_stnd_FD(mesh750_2,[],[],[],0,solver,opt);

    mesh850_2 = mesh850;
    mesh850_2.mua(region==1) = mesh850.mua(region==1)*(1+all_fluctuate2(1,rep));
    mesh850_2.mua(region==2) = mesh850.mua(region==2)*(1+all_fluctuate2(2,rep));
    mesh850_2.mua(region==3) = mesh850.mua(region==3)*(1+all_fluctuate2(3,rep));
    mesh850_2.mua(region==4) = mesh850.mua(region==4)*(1+all_fluctuate2(4,rep));
    mesh850_2.mua(region==5) = mesh850.mua(region==5)*(1+all_fluctuate2(5,rep));
    mesh850_2.mus(region==1) = mesh850.mus(region==1)*(1+all_fluctuate2(6,rep));
    mesh850_2.mus(region==2) = mesh850.mus(region==2)*(1+all_fluctuate2(7,rep));
    mesh850_2.mus(region==3) = mesh850.mus(region==3)*(1+all_fluctuate2(8,rep));
    mesh850_2.mus(region==4) = mesh850.mus(region==4)*(1+all_fluctuate2(9,rep));
    mesh850_2.mus(region==5) = mesh850.mus(region==5)*(1+all_fluctuate2(10,rep));
    mesh850_2.kappa = 1./(3*(mesh850_2.mua + mesh850_2.mus));
    J850 = jacobiangrid_stnd_FD(mesh850_2,[],[],[],0,solver,opt);

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

clear J750 J850
% save('tmp_spec4','-v7.3')

%% Reconstruction
fprintf("Reconstruction...\n")
mesh850 = load_mesh('JoeMeshCCW1/CCW1Mesh');


mesh850 = gen_intmat(mesh850, xgrid, ygrid, zgrid);

J850 = jacobiangrid_stnd_FD(mesh850,[],[],[],0,solver,opt);

mesh750 = mesh850;
region = mesh750.region;
mesh750.mua(region==2) = 0.0167;
mesh750.mua(region==3) = 0.0180;
mesh750.mua(region==4) = 0.0116;
mesh750.mua(region==5) = 0.0170;

mesh750.mus(region==2) = 1.1908;
mesh750.mus(region==3) = 0.8359;
mesh750.mus(region==4) = 0.94;
mesh750.mus(region==5) = 0.74;
mesh750.kappa = 1./(3*(mesh750.mua + mesh750.mus));
J750 = jacobiangrid_stnd_FD(mesh750,[],[],[],0,solver,opt);

load E.mat
% Apply beer-lambert law to get the "spectral" jacobian
J_spec = [[J750.complete(1:672,:)*E(1,1), J750.complete(1:672,:)*E(1,2)]; [J850.complete(1:672,:)*E(2,1), J850.complete(1:672,:)*E(2,2)]];
clear J750 J850
% This step uses a LOT of memory and can crash Matlab
% might be a good idea to save the results before running these lines
L = sqrt(0.1 + sum(J_spec.^2));
Ap = J_spec./L;
clear J_spec
[~, inv_op] = tikhonov(Ap, 0.01);
inv_op = inv_op./L';
clear Ap

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
    tmp = reshape(mesh750.vol.mesh2grid*all_beta(:,i), length(ygrid), length(xgrid), length(zgrid));
    tmp = tmp(y_idx,x_idx,z_idx);
    clean_images(:,:,:,i) = flip(tmp./std(tmp(:)), 2);
end

mask0=zeros(32,80,64);
mask0(mesh750.vol.gridinmesh)=1;
mask=mask0(y_idx,x_idx,z_idx);

fprintf("Saving...\n")
save('Datasets/images_CCW1Mesh_spec4_2.mat','clean_images','noisy_images','mask', '-v7.3');
save('Datasets/Data_CCW1Mesh_spec4_2.mat', 'all_recon', 'all_beta', 'all_x', 'all_y', 'all_z', 'all_r', 'all_amplitude', 'all_dOD', 'all_dOD_noisy', 'all_noise','all_ratio','all_fluctuate2','all_fluctuate1', 'samples', '-v7.3');

