% input Mag
Sigma = 1; 
[X,Y]   = ndgrid(-round(3*Sigma):round(3*Sigma));
DGaussxx = 1/(2*pi*Sigma^4) * (X.^2/Sigma^2 - 1) .* exp(-(X.^2 + Y.^2)/(2*Sigma^2));
DGaussxy = 1/(2*pi*Sigma^6) * (X .* Y)           .* exp(-(X.^2 + Y.^2)/(2*Sigma^2));
DGaussyy = DGaussxx';
Dxx = imfilter(Mag,DGaussxx,'conv');
Dxy = imfilter(Mag,DGaussxy,'conv');
Dyy = imfilter(Mag,DGaussyy,'conv');
DD = sqrt((Dxx - Dyy).^2 + 4*Dxy.^2);
mu1 = 0.5*(Dxx + Dyy + DD);
mu2 = 0.5*(Dxx + Dyy - DD);
check=abs(mu1)>abs(mu2);
Lambda1=mu1; Lambda1(check)=mu2(check);
Lambda2=mu2; Lambda2(check)=mu1(check);
% return Lambda1, Lambda2
