SST=ncread(filename,'sst_regression');
SST=double(SST(R_end:-1:R_start,C_start:C_end))';
Grad_thresh=0.3;
Edge_thresh=0.3;
std_thresh=1;
w_in =7;
w_out=9; %w_out=w_in+2;
w_g =21; %w_g=(w_out+1)*2+1;
h=[0.036420 0.248972 0.429217 0.248972 0.036420];
hp=[0.108415 0.280353 0 -0.280353 -0.108415];
[I,J]=size(SST);
dX = filter2(h',filter2(hp,SST));
dY = filter2(hp',filter2(h,SST));
Mag=sqrt(dX.^2+dY.^2);
[Dxx,Dxy,Dyy] = Hessian2D(Mag);
[Lam1,Lam2,Ix,Iy]=eig2image(Dxx,Dxy,Dyy);

D=SST-medfilt2(SST,[w_in w_in]);
easyclouds = zeros(I,J);
ind= find(SST<270 | Mag>Grad_thresh | abs(D)>Edge_thresh);
easyclouds(ind)=1;

easyfronts = zeros(I,J);
ind= find(SST > 270 & Mag > Grad_thresh & abs(D) < Edge_thresh & Lam2 < -0.05);
easyfronts(ind)=1;

ind=find(easyclouds==1 & easyfronts~=1);
MaskF=zeros(I,J);
MaskF(ind)=1;

Min_front_length=20;
CC = bwconncomp(MaskF,8);
count=1;
Mask_front=zeros(I,J);
for k=1:CC.NumObjects
    if length(CC.PixelIdxList{k})>Min_front_length
        Mask_front(CC.PixelIdxList{k})=count;
        count=count+1;
        else
        MaskF(CC.PixelIdxList{k})=0;
    end
end
