%%%%%%% THIS you should already have %%%%%%%%%%%%%%%%%%%%%%%

filename='ACSPO_V2.41_NPP_VIIRS_2012-02-07_1700-1710_20170726.221145.nc';
sst=double(ncread(filename,'sst_regression'))';
BT11=double(ncread(filename,'brightness_temp_chM15'))';
BT12=double(ncread(filename,'brightness_temp_chM16'))';
BT8=double(ncread(filename,'brightness_temp_chM14'))';
acspo_mask=ncread(filename,'acspo_mask')';

M10=ncread(filename,'albedo_chM10')'; % just for debugging
figure; z=imagesc(M10,[0 10]); colormap(gray); title('1.6 micron albedo');

[I,J]=size(sst);
ind=find(bitget(acspo_mask,7)==0 & bitget(acspo_mask,8)==0);
ACSM=zeros(I,J);
ACSM(ind)=1;
figure; z=imagesc(ACSM); set(z,'alphadata',isfinite(sst)); colormap(jet);

sst_low = 271.15;
sst_high = 310;
delta_n=0.1;
delta_Lam=0.05;
thresh_mag=0.2;
thresh_mag_low=0.1;
thresh_L=0.8;
median_thresh=0.5;
grad_ratio_thresh=0.5;

h=[0.036420 0.248972 0.429217 0.248972 0.036420];
hp=[0.108415 0.280353 0 -0.280353 -0.108415];

dX = -filter2(h',filter2(hp,sst));
dY = filter2(hp',filter2(h,sst));
gradmag=sqrt(dX.^2+dY.^2);

dXdX = -filter2(h',filter2(hp,dX));
dYdY = filter2(hp',filter2(h,dY));
L=dXdX.^2+dYdY.^2;
 
[Dxx,Dxy,Dyy] = Hessian2D(gradmag);
[Lam1,Lam2,Ix,Iy]=eig2image(Dxx,Dxy,Dyy);

dX_BT11 = -filter2(h',filter2(hp,BT11));
dY_BT11 = filter2(hp',filter2(h,BT11));
gradmag_BT11=sqrt(dX_BT11.^2+dY_BT11.^2);

dX_BT12 = -filter2(h',filter2(hp,BT12));
dY_BT12 = filter2(hp',filter2(h,BT12));
gradmag_BT12=sqrt(dX_BT12.^2+dY_BT12.^2);

MF = medfilt2(sst, [5 5]);

COLD_CLOUD = -9;
NN_test=-8;
BT12_test=-7;
TEST_LAPLACIAN = -6;
RATIO_TEST=-5;
TEST_UNIFORMITY = -4;
TEST_CLOUD_BOUNDARY = -3;
TEST_LAPLACIAN_HIST = -2;
TEST_LAPLACIAN_HIST_FRONT = -1;
TEST_GRADMAG_LOW = 0;
TEST_LOCALMAX = 1;
FRONT_GUESS = 2;

Mask=zeros(I,J);
ind=find(gradmag<thresh_mag); Mask(ind)=TEST_GRADMAG_LOW;
ind=find(sst<BT12); Mask(ind)=BT12_test;
%ind=find(logmag>thresh_mag); Mask(ind)=FRONT_GUESS;
ind=find(gradmag>thresh_mag); Mask(ind)=FRONT_GUESS;
ind=find(gradmag-gradmag_BT11<-delta_n); Mask(ind)=TEST_CLOUD_BOUNDARY;
ind=find(L>thresh_L); Mask(ind)=TEST_LAPLACIAN;
ind=find(sst<sst_low); Mask(ind)=COLD_CLOUD;
ind=find(Lam2>-delta_Lam & Mask==FRONT_GUESS); Mask(ind)=TEST_LOCALMAX;
ind=find(abs(sst-MF)>median_thresh); Mask(ind)=TEST_UNIFORMITY;
ind_grad=find(gradmag>thresh_mag);
ind=find((gradmag(ind_grad)-gradmag_BT11(ind_grad))./gradmag(ind_grad)>grad_ratio_thresh); Mask(ind_grad(ind))=RATIO_TEST;
%ind=find(Mask==FRONT_GUESS & eigIm>eig_thresh); Mask(ind)=EIG_TEST; 

figure; z=imagesc(Mask); colormap(jet); set(z,'alphadata',isfinite(sst)); title('Mask based on constant thresholds');

%%%%%%% THIS you should already have %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%% COMPRING to ACSPO and labeling agree/disagree

Label=zeros(I,J);
ind=find(Mask<0); Label(ind)=1;
ind_o=find(isfinite(sst)==1 & Label==0 & ACSM==1);     % easyclouds: ocean; acspo: ocean
ind_cl=find(isfinite(sst)==1 & Label==1 & ACSM==0);    % easyclouds: cloud, acspo: cloud
ind_test=find(isfinite(sst)==1 & Label==0 & ACSM==0);  % easyclouds: ocean, acspo: cloud
Mask_train=zeros(I,J);
Mask_train(ind_cl)=1;
Mask_train(ind_o)=2;
Mask_train(ind_test)=3;
figure; z=imagesc(Mask_train); set(z,'alphadata',isfinite(sst)); colormap(jet);

%%%%%%% COMPRING to ACSPO and labeling agree/disagree


%%%%% NEW THING: using nearest neighbor search.
% Matlab has training and testing in one call (kinda inconvenient, but not
% crusially important here)

alpha=10;
[ICX,C]=knnsearch([BT11(ind_o) alpha*(BT11(ind_o)-BT12(ind_o)) alpha*(BT11(ind_o)-BT8(ind_o))],[BT11(ind_test) alpha*(BT11(ind_test)-BT12(ind_test)) alpha*(BT11(ind_test)-BT8(ind_test))]);
ind=find(C>0.1);
Mask_train(ind_test(ind))=1;
z=imagesc(Mask_train); set(z,'alphadata',isfinite(sst)); colormap(jet);

Mask(ind_test(ind))=NN_test;  % adding the NN flag to the front mask
figure; z=imagesc(Mask); set(z,'alphadata',isfinite(sst)); colormap(jet); title('After NN test')

% JUST for debugging

ind=find(Mask_train==2 | Mask_train==3);
Q=NaN*ones(I,J);
Q(ind)=1;

figure
subplot(1,2,1); z=imagesc(M10.*Q,[0 10]); set(z,'alphadata',isfinite(sst)); colormap(cool);
subplot(1,2,2); z=imagesc(M10.*ACSM,[0 10]); set(z,'alphadata',isfinite(sst)); colormap(cool);


