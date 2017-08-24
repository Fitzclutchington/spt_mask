F=zeros(size(fronts));
indF=find(fronts==0 | fronts==1 | fronts==2); % to get your Init fronts
F(indF)=1;

% you have dX and dY
h=[0.036420 0.248972 0.429217 0.248972 0.036420];
hp=[0.108415 0.280353 0 -0.280353 -0.108415];
dX = filter2(h',filter2(hp,sst));
dY = filter2(hp',filter2(h,sst));

% the actuall connecting portion:
w=10; 
G=F;
[I,J]=size(sst);
ind_invalid=find(gradmag<0.05 | easyclouds_new>0);
ddX=dX;
ddY=dY;
ddX(ind_invalid)=NaN; ddY(ind_invalid)=NaN;

for k=1:length(indF)
    i=mod(indF(k)-1,I)+1;
    j=ceil(indF(k)/I);
    if i>w & j>w & i < I-w & j < J-w
    dx_w=ddX(i-w:i+w,j-w:j+w);
    dy_w=ddY(i-w:i+w,j-w:j+w);
    z=[dx_w(:) dy_w(:)];
    d=z*[dX(i,j), dY(i,j)]';
    [dm,jm]=nanmax(d);
    if length(jm)>=1
        i_new=mod(jm-1,2*w+1)+1;
        j_new=ceil(jm/(2*w+1));
        G(i-w+i_new-1,j-w+j_new-1)=1;
        ddX(i-w+i_new-1,j-w+j_new-1)=NaN;
        ddY(i-w+i_new-1,j-w+j_new-1)=NaN;
    end
    end
end

% this is to threshold small fronts and to determine the sides
% the current way of determining the sides may be just fine
min_front_length=100;
CC = bwconncomp(G,8);
Mask=G;
for k=1:CC.NumObjects
    ind_k=CC.PixelIdxList{k};
    if length(ind_k)<min_front_length
        Mask(ind_k)=0;
    else
        [IDX,otsu_thresh,cluster_means]=otsu(sst(ind_k));
        Mask(ind_k)=IDX;
    end
 end