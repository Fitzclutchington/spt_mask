filename='20160517061000-STAR-L2P_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.41B07-v02.0-fv01.0.nc';
sst=double(ncread(filename,'sea_surface_temperature'))';
BT11=double(ncread(filename,'brightness_temperature_11um'))';
BT12=double(ncread(filename,'brightness_temperature_12um'))';
% lat=double(ncread(filename,'lat'))';
% lon=double(ncread(filename,'lon'))';

filename='sstmag.nc';
logmag=ncread(filename,'data')';

[I,J]=size(sst);
sst_low = 271.15;
sst_high = 310;
delta_n=0.1;
delta_Lam=0.05;
thresh_mag=0.2;
thresh_L=0.8;
dist_thresh=0.5;
grad_ratio=2.5;
uniformity_thresh=0.5;
theta_interp=0.2;
%theta_U=1;


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

COLD_CLOUD = -7;
TEST_LAPLACIAN = -6;
RATIO_TEST=-5;
TEST_UNIFORMITY = -4;
TEST_CLOUD_BOUNDARY = -3;
TEST_LAPLACIAN_HIST = -2;
%TEST_UNIFORMITY_HIST = -1.5;
TEST_LAPLACIAN_HIST_FRONT = -1;
%TEST_UNIFORMITY_HIST_FRONT = -0.5;
TEST_GRADMAG_LOW = 0;
TEST_LOCALMAX = 1;
FRONT_GUESS = 2;

Mask=zeros(I,J);
ind=find(gradmag<thresh_mag); Mask(ind)=TEST_GRADMAG_LOW;
ind=find(logmag>thresh_mag); Mask(ind)=FRONT_GUESS;
ind=find(gradmag>thresh_mag & gradmag_BT11-gradmag_BT12<-delta_n); Mask(ind)=TEST_CLOUD_BOUNDARY;
ind=find(L>thresh_L); Mask(ind)=TEST_LAPLACIAN;
ind=find(sst<sst_low); Mask(ind)=COLD_CLOUD;
ind=find(Lam2>-delta_Lam & Mask==FRONT_GUESS); Mask(ind)=TEST_LOCALMAX;
ind=find(Mask==FRONT_GUESS & gradmag./eigIm>grad_ratio); Mask(ind)=RATIO_TEST;

%%%% Uniformity image used to further mask histogram
h = fspecial('gaussian', 7, 1);
B = imfilter(sst,h);
ind=find(abs(sst-B)>uniformity_thresh); Mask(ind)=TEST_UNIFORMITY;
%%%%%%%%%

figure; z=imagesc(Mask); colormap(jet); set(z,'alphadata',isfinite(sst)); title('Mask before hist check');

step = 10;
d_sst = 0.2;

y_size = ceil(I/step)+1;
x_size = ceil(J/step)+1;
sst_range = ceil(sst_high - sst_low);
sst_size = ceil(sst_range/d_sst) + 1;

HL=zeros(y_size,x_size,sst_size);
% HU=zeros(y_size,x_size,sst_size);
for y = 1:I
    for x = 1:J
        if(Mask(y,x)==TEST_LAPLACIAN & isfinite(sst(y,x))==1)
            m = round(y/step)+1;
            n = round(x/step)+1;
            s = round((sst(y,x)-sst_low)/d_sst)+1;
            
            if s < 1
                s = 1;
            end
            if s > sst_size
                s = sst_size;
            end
            
            for i = -1:1
                for j = -1:1
                    for k = -s+1:0
                        if(n+j >= 1 & m+i>=1 & n+j <= x_size & m+i <= y_size & s+k < sst_size )
                            HL(m+i,n+j,s+k)=HL(m+i,n+j,s+k)+1;
                        end
                    end
                end
            end
        end
        
%         if(abs(BT11(y,x)-B(y,x))>theta_U & isfinite(sst(y,x))==1)
%             m = round(y/step)+1;
%             n = round(x/step)+1;
%             s = round((sst(y,x)-sst_low)/d_sst)+1;
%             
%             if s < 1
%                 s = 1;
%             end
%             if s > sst_size
%                 s = sst_size;
%             end
%             
%             for i = -1:1
%                 for j = -1:1
%                     for k = -s+1:0
%                         if(n+j >= 1 & m+i>=1 & n+j <= x_size & m+i <= y_size & s+k < sst_size )
%                             HU(m+i,n+j,s+k)=HU(m+i,n+j,s+k)+1;
%                         end
%                     end
%                 end
%             end
%         end
        
    end
end

p=1;
for y = 1:I
    for x = 1:J
        if( Mask(y,x)==FRONT_GUESS & isfinite(sst(y,x))==1)
            m = round(y/step)+1;
            n = round(x/step)+1;
            s = round((sst(y,x)-sst_low)/d_sst)+1;
            
            if s < 1
                s = 1;
            end
            if s > sst_size
                s = sst_size;
            end
            
            if(n >= 1 & m>=1 & n <= x_size & m <= y_size)
                if HL(m,n,s)>p
                    Mask(y,x)=TEST_LAPLACIAN_HIST_FRONT;
%                 else if HU(m,n,s)>p
%                         Mask(y,x)=TEST_UNIFORMITY_HIST_FRONT;
%                     end
                end
            end
        end
        
        if( Mask(y,x)==TEST_GRADMAG_LOW & isfinite(sst(y,x))==1)
            m = round(y/step)+1;
            n = round(x/step)+1;
            s = round((sst(y,x)-sst_low)/d_sst)+1;
            
            if s < 1
                s = 1;
            end
            if s > sst_size
                s = sst_size;
            end
            
            if(n >= 1 & m >= 1 & n <= x_size & m <= y_size )
                if HL(m,n,s)>p
                    Mask(y,x)=TEST_LAPLACIAN_HIST;
%                 else if HU(m,n,s)>p
%                         Mask(y,x)=TEST_UNIFORMITY_HIST;
%                     end     
                end
            end
            
        end
    end
end

figure; z=imagesc(Mask); colormap(jet); set(z,'alphadata',isfinite(sst)); title('Mask after HL hist check');

%%%% Prototype for use of uniformity image
Label=zeros(I,J);
ind=find(abs(B-sst)>theta_interp);
Label(ind)=1;
ind=find(abs(B-sst)<theta_interp);
Label(ind)=-1;

w=5; f = fspecial('gaussian',2*w+1,w/3);
sst_cl_ref=NaN*ones(I,J); 
p_w=0.1;

for y = 1:I
    for x = 1:J
        if Label(y,x)==1 & Mask(y,x) >=0 & y-w>=1 & y+w<=I & x-w>=1 & x+w<=J
            win=sst(y-w:y+w,x-w:x+w);
            win_label=Label(y-w:y+w,x-w:x+w);
            ind_o=find(win_label==-1);
            if length(ind_o)>p_w*(2*w+1)^2
                sst_cl_ref(y,x)=sum(win(ind_o).*f(ind_o))/sum(f(ind_o));
            end
        end
    end
end
ind=find(Label==-1);
sst_cl_ref(ind)=sst(ind);

figure; z=imagesc(sst_cl_ref,[270 305]); colormap(jet);
ind=find(Mask<0);
Label=ones(I,J);
Label(ind)=NaN;
set(z,'alphadata',isfinite(Label+sst));

dX_cl_ref = -filter2(h',filter2(hp,sst_cl_ref));
dY_cl_ref = filter2(hp',filter2(h,sst_cl_ref));
gradmag_cl_ref=sqrt(dX_cl_ref.^2+dY_cl_ref.^2);

figure; z=imagesc(gradmag_cl_ref,[0 0.5]); set(z,'alphadata',isfinite(sst_cl_ref)); colormap(cool);
%%%%%%%%%