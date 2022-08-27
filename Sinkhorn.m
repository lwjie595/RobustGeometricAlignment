function [EMD,FM]=Sinkhorn(TM_A,B,WA,WB,tau)
% Squared Euclidean distance matrix

disMat=distance(TM_A,B);
disMat(disMat<0)=0;
if tau>0
    A_add=tau*sum(WA);
    B_add=tau*sum(WB);
    WA=[WA A_add];
    WB=[WB B_add];
    maxdist=max(max(disMat,[],1))*10;
    disMat=[disMat zeros(size(disMat,1),1)];
    disMat=[disMat;zeros(1,size(disMat,2))];
    disMat(end,end)=maxdist;
end

 lambda=10/median(disMat(:));
K=exp(-lambda*disMat);
U=K.*disMat;
[EMD,~,u,v]=Transport(WA',WB',K,U,lambda);
FM=bsxfun(@times,v',bsxfun(@times,u,K));
EMD=EMD/(1-tau);
if tau>0
FM=FM(1:end-1,1:end-1);
end


end


