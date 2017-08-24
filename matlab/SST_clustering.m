filename='R20140709_ACSPO_V2.30_NPP_VIIRS_2014-06-20_1710-1719_20140623.071032.nc';
[M15,Cloud_Acspo_resampled,Lat,Lon]=resample(filename,'brightness_temp_chM15');
[M16,Cloud_Acspo_resampled,Lat,Lon]=resample(filename,'brightness_temp_chM16');
[SST,Cloud_Acspo_resampled,Lat,Lon]=resample(filename,'sst_regression');
[I,J]=size(SST);

h=[0.036420 0.248972 0.429217 0.248972 0.036420];
hp=[0.108415 0.280353 0 -0.280353 -0.108415];

t_step=2;
td_step=0.5;
Low_mag=0.1;
High_mag=0.5;
T0=270;
N_comp=200;

DT=M15-M16;
dX = filter2(h',filter2(hp,SST));
dY = filter2(hp',filter2(h,SST));
Mag=sqrt(dX.^2+dY.^2);
%Cos_Theta=cos(atan2(dX,dY));
%ind=find(Mag<Low_mag);
%Cos_Theta(ind)=NaN;

%[Dxx,Dxy,Dyy] = Hessian2D(Mag);
%[Lam1,Lam2,Ix,Iy]=eig2image(Dxx,Dxy,Dyy);

indW=find(Mag<Low_mag & SST>T0 & DT>-0.5);
T=-ones(I,J);
T(indW)=round((SST(indW)-T0)/t_step);
TD=-ones(I,J);
TD(indW)=round((DT(indW)+1)/td_step);

H=zeros(I*J,4);
for i=0:max(T(:))
    for j=0:max(TD(:))
        ind=find(T==i & TD==j);
        MaskIJ=zeros(size(SST));
        MaskIJ(ind)=1;
        CC = bwconncomp(MaskIJ,8);
        for k=1:CC.NumObjects
            inds=CC.PixelIdxList{k};
            L=length(inds);
            if L>=N_comp
                H(inds,:)=[Lat(inds) Lon(inds) nanmean(SST(inds))*ones(L,1) nanmean(DT(inds))*ones(L,1) nanmean(anomaly(inds))];
            end
        end
    end
end

indC=find(SST>T0);
ind=find(H(:,1)~=0);
[IDX,D] = knnsearch(H(ind,:),[Lat(indC) Lon(indC) SST(indC) DT(indC)]);
MaskID=zeros(I,J);
MaskID(indC)=H(ind(IDX),3);
