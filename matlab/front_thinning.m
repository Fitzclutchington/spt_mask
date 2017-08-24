[I,J]=size(sst);
MaskI=zeros(I,J);
ind=find(fronts==2 | fronts==0);
MaskI(ind)=1;
CI = bwconncomp(MaskI,8);
Min_front_length=50;
alpha=-5:5;
N_alpha=length(alpha);
W=zeros(I,J);
Fr_length=zeros(I,J);
Fr_strength=zeros(I,J);
for k=1:CI.NumObjects
    ind=CI.PixelIdxList{k};
    L_fr=length(CI.PixelIdxList{k});
    if L_fr>Min_front_length
        i=mod(ind-1,I)+1;
        j=ceil(ind/I);
        iB=repmat(i,1,N_alpha)-repmat(alpha,L_fr,1).*repmat(dY(ind)./gradmag(ind),1,N_alpha);
        jB=repmat(j,1,N_alpha)+repmat(alpha,L_fr,1).*repmat(dX(ind)./gradmag(ind),1,N_alpha);
        indB=round(jB)*I+round(iB);
        for m=1:L_fr
            [T_max, j_max]=max(sstmag(indB(m,:)));
            W(indB(m,j_max))=1;
        end
        Fr_length(ind)=sum(W(ind));
        Fr_strength(ind)=nanmean(sstmag(ind).*W(ind));
    end
end
