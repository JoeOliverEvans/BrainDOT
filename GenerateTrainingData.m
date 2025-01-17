addpath(genpath('../MatlabPackages'))
clear


load('log.mat')    
clear info

%% Load Mesh and extinction coefficients and setup grid
mesh = load_mesh('Mesh/mesh');
xgrid = -79:2:79;
ygrid = -119:2:-57;
zgrid = -58:2:68;
mesh = gen_intmat(mesh, xgrid, ygrid, zgrid);
gray = mesh.region==3;
N_gray = sum(gray);
gray_coord = mesh.nodes(gray,:);
N_nodes = size(mesh.nodes,1);


load E.mat % File containing extinction coefficients of HbO and Hb supplied by Neurodot for 750 and 850 nm

%% Set variables
samples = 3000;
all_amplitude = [0.0004 * rand(samples, 1)+ 0.0001, 0 * ones(samples, 1)]; % amplitude between 0.0001 and 0.0005
all_ratio = rand(samples)*2 + 3;%rand(samples)*1.5 + 2.5;

% randomize the radius of the activation spot
all_r = rand(samples,2)*10 + 5; % between 5 and 15mm

all_fluctuate1 = 0.2 .* rand(10, samples) - 0.1; % Centre the fluctuations around 0  10% either side


% These will be the centers of the activation spot
all_x = zeros(2, samples);
all_y = zeros(2, samples);
all_z = zeros(2, samples);

% For storing values
all_dOD = zeros(1344, samples);
all_beta = zeros(N_nodes, samples);

%% Main loop
for rep=1:samples
    fprintf('%d/%d\n', rep, samples);

    % Calculate Sensitivity Profile
    % Set flags
    flags.tag=[padname,'_on',meshname,'_test'];
    flags.gridname=gridname;
    flags.meshname=meshname;
    flags.head='info';
    flags.info=infoT1;                  % Your T1 info file
    flags.gthresh=1e-5;                 % Voxelation threshold in G
    flags.voxmm=2;                      % Voxelation resolution (mm)
    flags.labels.r1='csf';              % Regions for optical properties
    flags.labels.r2='white';
    flags.labels.r3='gray';
    flags.labels.r4='bone';
    flags.labels.r5='skin';
    flags.op.lambda=[750,850];          % Wavelengths (nm)
    flags.op.mua_skin=[0.0170,0.0190].*[all_fluctuate1(1,rep), all_fluctuate1(6,rep)];  % Baseline absorption
    flags.op.mua_bone=[0.0116,0.0139].*[all_fluctuate1(2,rep), all_fluctuate1(7,rep)];
    flags.op.mua_csf=[0.0040,0.0040].*[all_fluctuate1(3,rep), all_fluctuate1(8,rep)];
    flags.op.mua_gray=[0.0180,0.0192].*[all_fluctuate1(4,rep), all_fluctuate1(9,rep)];
    flags.op.mua_white=[0.0167,0.0208].*[all_fluctuate1(5,rep), all_fluctuate1(10,rep)];
    flags.op.musp_skin=[0.74,0.64];     % Baseline reduced scattering coeff
    flags.op.musp_bone=[0.94,0.84];
    flags.op.musp_csf=[0.3,0.3];
    flags.op.musp_gray=[0.8359,0.6726];
    flags.op.musp_white=[1.1908,1.0107];
    flags.op.n_skin=[1.4,1.4];          % Index of refraction
    flags.op.n_bone=[1.4,1.4];
    flags.op.n_csf=[1.4,1.4];
    flags.op.n_gray=[1.4,1.4];
    flags.op.n_white=[1.4,1.4];
    flags.srcnum=Ns;                    % Number of sources
    flags.t4=affine_Subj2MNI;           % Affine matrix for going from subject-specific space to MNI space
    flags.t4_target='MNI'; % string
    flags.makeA=1; % don't make A, just make G
    flags.Hz=0;
    if flags.Hz, flags.tag = [flags.tag,'FD']; end
    
    flags.joenVtoggle = true;
    
    % Run makeAnirfast to get sensitivity matrix
    Ti=tic;[A,dim,Gsd]=makeAnirfaster(mesh,flags); % size(A)= [Nwl, Nmeas, Nvox]
    disp(['<makeAnirfast took ',num2str(toc(Ti))])
    
    % Package data and save A
    [Nwl,Nmeas,Nvox]=size(A);
    A=reshape(permute(A,[2,1,3]),Nwl*Nmeas,Nvox);
    
    % Place spatial information about light model in info.tissue structure
    info.tissue.dim=dim;
    info.tissue.affine=flags.t4;
    info.tissue.infoT1=infoT1;
    info.tissue.affine_target='MNI';
    info.tissue.flags=flags;
    

    if length(size(A))>2  % A data structure [wl X meas X vox]-->[meas X vox]
        [Nwl,Nmeas,Nvox]=size(A);
        A=reshape(permute(A,[2,1,3]),Nwl*Nmeas,Nvox);
    end  

    centers = [-55 + (55--55) .* rand([1,2]);[-85, -85];-15 + (40--15).*rand([1,2])]';
    tol = 2;
    nodes_ingray_intol = gray_coord(squeeze(gray_coord(:, 1)>centers(1,1)-tol & gray_coord(:,1)<centers(1,1)+tol & gray_coord(:,3)>centers(1,3)-tol & gray_coord(:,3)<centers(1,3)+tol), 2);
    sizea = size(nodes_ingray_intol);
    if sizea(1) ~= 0
        if min(nodes_ingray_intol) < -75
            % changes the y value to the new value if it is truly in about
            % the right spot
            centers(1,2) = min(nodes_ingray_intol);
        else
            fprintf('uh oh\n');
        end
    end

    all_x(:,rep) = centers(:,1);
    all_y(:,rep) = centers(:,2);
    all_z(:,rep) = centers(:,3);
    
    % The dHbO vector in source space
    beta = zeros(N_nodes,1);
    betavis = zeros(N_nodes,1);
    for i=1:2
        idx = vecnorm(mesh.nodes-centers(i,:),2,2)<all_r(rep) & gray;
        beta(idx) = all_amplitude(rep,i);
    end
    all_beta(:,rep) = beta;

    %%% Calculate measurements
    all_ratio = 3;
    
    beta1 = (E(1,1)*beta-E(1,2)*beta/all_ratio);
    %figure;scatter3(mesh.nodes(find(mesh.region==3), 1), mesh.nodes(find(mesh.region==3), 2), mesh.nodes(find(mesh.region==3), 3),[], beta1(find(mesh.region==3)));axis equal;
    
    beta1_shaped = mesh.vol.mesh2grid*beta1; % from mesh space to voxel space
    beta1_shaped = reshape(beta1_shaped, length(ygrid), length(xgrid), length(zgrid)); % reshape to the voxel volume
    beta1_shaped = permute(beta1_shaped, [2,1,3]); % swap x and y as when coverting to voxel space x and y are swapped (graphs vs images) and this is what this function is made for
    beta1_shaped = flip(beta1_shaped, 1); % convert from left handed to right handed coordinate
    beta1_shaped = reshape(beta1_shaped, [length(ygrid)* length(xgrid)* length(zgrid), 1]); % flatten to use with A matrix
    
    
    beta2 = (E(2,1)*beta-E(2,2)*beta/all_ratio);
    %figure;scatter3(mesh.nodes(find(mesh.region==3), 1), mesh.nodes(find(mesh.region==3), 2), mesh.nodes(find(mesh.region==3), 3),[], beta2(find(mesh.region==3)));axis equal;
    
    beta2_shaped = mesh.vol.mesh2grid*beta2;
    beta2_shaped = reshape(beta2_shaped, length(ygrid), length(xgrid), length(zgrid));
    beta2_shaped = permute(beta2_shaped, [2,1,3]);
    beta2_shaped = flip(beta2_shaped, 1);
    beta2_shaped = reshape(beta2_shaped, [length(ygrid)* length(xgrid)* length(zgrid), 1]);
    
    all_dOD(:, rep) = [A(1:end/2, :) * beta1_shaped(info.tissue.dim.Good_Vox, :); A(end/2+1:end, :) * beta2_shaped(info.tissue.dim.Good_Vox, :)]; % deal with the wavlengths separately
end
clear A

load('CroppedWithjoenVtoggle/A_AdultV24x28_onHD_Mesh0_test.mat');

%%
noise_percent = repmat([NoiseFunction(info.pairs.r2d(1:end/2),0,750); NoiseFunction(info.pairs.r2d(end/2+1:end),0,850)], 1,samples); % Gives the standard deviation of that gausian noise per channel length and wavlength
noise_std = noise_percent/100; % divide by 100 as the value given by the function is actually a percentage?
k = 1+sum((noise_std) .* randn([size(all_dOD, 1), size(all_dOD, 2), 40]), 3)/40; % going from a noise model at 40Hz to one at 1Hz resample as per the real data (forgetting about block averaging)
all_dOD_noisy = all_dOD + log(k); 


%% Reconstruction
% Load A matrix

if length(size(A))>2  % A data structure [wl X meas X vox]-->[meas X vox]
    [Nwl,Nmeas,Nvox]=size(A);
    A=reshape(permute(A,[2,1,3]),Nwl*Nmeas,Nvox);
end    

fprintf("Reconstruction...\n")
Nvox=size(A,2);
Nt=size(all_dOD,2);
cortex_mu_a=zeros(Nvox,Nt,2);
cortex_mu_a_smoothed=zeros(Nvox,Nt,2);

for j = 1:2
    keep = (info.pairs.WL == j) & (info.pairs.r2d <= 47);
    disp('> Inverting A')                
    iA = Tikhonov_invert_Amat(A(keep, :), 0.01, 0.1); % Invert A-Matrix
    disp('> Smoothing iA')
    iA_smoothed = smooth_Amat(iA, info.tissue.dim, 3);         % Smooth Inverted A-Matrix      
    cortex_mu_a(:, :, j) = reconstruct_img(all_dOD_noisy(keep,:), iA);% Reconstruct Image Volume
    cortex_mu_a_smoothed(:, :, j) = reconstruct_img(all_dOD_noisy(keep,:), iA_smoothed);
end

% Spectroscopy
if ~exist('E', 'var'),load('E.mat'),end
cortex_Hb = spectroscopy_img(cortex_mu_a, E);
cortex_Hb_smoothed = spectroscopy_img(cortex_mu_a_smoothed, E);
cortex_HbO = cortex_Hb(:, :, 1);
cortex_HbO_smoothed = cortex_Hb_smoothed(:, :, 1);

HbOvol = Good_Vox2vol(cortex_HbO,info.tissue.dim);
HbOvol_smoothed = Good_Vox2vol(cortex_HbO_smoothed,info.tissue.dim);

ground_truth = reshape(mesh.vol.mesh2grid*all_beta, length(ygrid), length(xgrid), length(zgrid), samples);
ground_truth = permute(ground_truth, [2,1,3,4]);
ground_truth = flip(ground_truth, 1);

save('TrainingData.mat', 'all_dOD', 'all_dOD_noisy', 'HbOvol', 'HbOvol_smoothed', 'all_beta', 'ground_truth', 'all_x', 'all_y', 'all_z', 'all_amplitude', 'all_fluctuate1', '-v7.3')