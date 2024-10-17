addpath(genpath('~/Documents/MATLAB/NeuroDOT'))
load('Pad_AdultV24x28.mat')

% info2 = info;
% tmp = info.optodes.spos3;
% src3 = [tmp(1:3,:);tmp(7:9,:);tmp(13:15,:);tmp(19:21,:)];
% tmp = info.optodes.spos2;
% src2 = [tmp(1:3,:);tmp(7:9,:);tmp(13:15,:);tmp(19:21,:)];
% tmp = info.optodes.dpos3;
% det3 = [tmp(1:4,:);tmp(9:12,:);tmp(17:20,:);tmp(25:28,:)];
% tmp = info.optodes.dpos2;
% det2 = [tmp(1:4,:);tmp(9:12,:);tmp(17:20,:);tmp(25:28,:)];
% 
% info2.optodes.spos2 = src2;
% info2.optodes.spos3 = src3;
% info2.optodes.dpos2 = det2;
% info2.optodes.dpos3 = det3;
% info2.optodes.CapName = 'Adult24x28Downsampled';
% 
% link = [];
% all_d2 = [];
% all_d3 = [];
% for i=1:length(src3)
%     for j=1:length(det3)
%         d3 = norm(src3(i,:) - det3(j,:));
%         if d3<40
%             link = [link; [i,j]];
%             all_d3 = [all_d3; d3];
%             all_d2 = [all_d2; norm(src2(i,:) - det2(j,:))];
%         end
%     end
% end
% info2.pairs.Src = link(:,1);
% info2.pairs.Det = link(:,2);
% info2.pairs.r2d = all_d2;
% info2.pairs.r3d = all_d3;
% info2.pairs.lambda = 850*ones(length(link),1);
% info2.pairs.NN = ones(length(link),1);
% info2.pairs.WL = ones(length(link),1);
% info2.pairs.Mod = info2.pairs.Mod(1:length(link));
% 
% info = info2;
% save('Pad_28x24Downsampled','info');

info2 = info;
src3 = info.optodes.spos3;
src2 = info.optodes.spos2;
det3 = info.optodes.dpos3;
det2 = info.optodes.dpos2;

link = [];
all_d2 = [];
all_d3 = [];
for i=1:length(src3)
    for j=1:length(det3)
        d3 = norm(src3(i,:) - det3(j,:));
        if d3<40
            link = [link; [i,j]];
            all_d3 = [all_d3; d3];
            all_d2 = [all_d2; norm(src2(i,:) - det2(j,:))];
        end
    end
end
info2.pairs.Src = link(:,1);
info2.pairs.Det = link(:,2);
info2.pairs.r2d = all_d2;
info2.pairs.r3d = all_d3;
info2.pairs.lambda = 850*ones(length(link),1);
info2.pairs.NN = ones(length(link),1);
info2.pairs.WL = ones(length(link),1);
info2.pairs.Mod = info2.pairs.Mod(1:length(link));

info = info2;
save('Pad_28x24Mod','info');
%% Create the mesh; copied form NeuroDOT example
[mask,infoT1]=LoadVolumetricData(['Segmented_MNI152nl_on_MNI111'],[],'4dfp');
% Parameters for generating your mesh
meshname=['Example_Mesh'];      % Provide a name for your mesh name here
param.facet_distance=1;   % Node position error tolerance at boundary
param.facet_size =1;      % boundary element size parameter
param.cell_size=1.5;        % Volume element size parameter
param.info=infoT1;
param.Offset=[0,0,0];
param.r0=5;                  % nodes outside of mask must be set to scalp==5;
param.CheckMeshQuality=0;
param.Mode=0;
tic;mesh=NirfastMesh_Region(mask,meshname,param);toc
pM.orientation='coord';pM.Cmap.P='gray';
% Put coordinates back in true space
mesh.nodes=change_space_coords(mesh.nodes,infoT1,'coord');

%% Place the optodes on the head mesh
tpos=cat(1,info.optodes.spos3,info.optodes.dpos3);
Ns=size(info.optodes.spos3,1);
Nd=size(info.optodes.dpos3,1);
rad=info;

% Adjust parameters by hand, update position, check
tpos2=tpos;
dx=0;
dy=-110;
dz=-55;
dS=1.1;
dxTh=-90;
dyTh=0;
dzTh=0;

tpos2=rotate_cap(tpos2,[dxTh,dyTh,dzTh]);
tpos2(:,1)=tpos2(:,1)+dx;
tpos2(:,2)=tpos2(:,2)+dy;
tpos2(:,3)=tpos2(:,3)+dz;
tpos2=scale_cap(tpos2,dS);

m0.nodes=mesh.nodes;
m0.elements=mesh.elements;
spos3=tpos2(1:Ns,:);
dpos3=tpos2((Ns+1):end,:);
tposNew=gridspringfit_ND2(m0,rad,spos3,dpos3);

PlotMeshSurface(mesh,pM);PlotSD(tposNew(1:Ns,:),tposNew((Ns+1):end,:),'render',gcf);

info.optodes.spos3=tposNew(1:Ns,:);
info.optodes.dpos3=tposNew((Ns+1):end,:);
info.tissue.infoT1=infoT1;
info.tissue.affine=eye(4);
info.tissue.affine_target='MNI';
save('Pad_28x24Mod_on_mesh.mat','info','tposNew'); % Pad file

%% Finish the mesh
source = [];
source.coord = info.optodes.spos3;
source.fixed = 0;
source.fwhm = zeros(Ns,1);
source.num = (1:Ns)';

detector = [];
detector.coord = info.optodes.dpos3;
detector.fixed = 0;
detector.fwhm = zeros(Nd,1);
detector.num = (1:Nd)';

mesh.source = source;
mesh.meas = detector;
mesh.link = [info.pairs.Src, info.pairs.Det, ones(length(info.pairs.Src),1)];

% Use only 850nm
% csf
idx = mesh.region==1;
mesh.mua(idx) = 0.0040;
mesh.mus(idx) = 0.3;
% white
idx = mesh.region==2;
mesh.mua(idx) = 0.0208;
mesh.mus(idx) = 1.0107;
% gray
idx = mesh.region==3;
mesh.mua(idx) = 0.0192;
mesh.mus(idx) = 0.6726;
% bone
idx = mesh.region==4;
mesh.mua(idx) = 0.0139;
mesh.mus(idx) = 0.84;
% skin
idx = mesh.region==5;
mesh.mua(idx) = 0.0190;
mesh.mus(idx) = 0.64;

mesh.ri = 1.4*ones(size(mesh.ri));
save_mesh('Example_Mesh', mesh);
% mesh = load_mesh('Example_Mesh');

%% Make the 750nm model
% csf
idx = mesh.region==1;
mesh.mua(idx) = 0.0040;
mesh.mus(idx) = 0.3;
% white
idx = mesh.region==2;
mesh.mua(idx) = 0.0167;
mesh.mus(idx) = 1.1908;
% gray
idx = mesh.region==3;
mesh.mua(idx) = 0.0180;
mesh.mus(idx) = 0.8359;
% bone
idx = mesh.region==4;
mesh.mua(idx) = 0.0116;
mesh.mus(idx) = 0.94;
% skin
idx = mesh.region==5;
mesh.mua(idx) = 0.0170;
mesh.mus(idx) = 0.74;

save_mesh('Example_Mesh750', mesh);

