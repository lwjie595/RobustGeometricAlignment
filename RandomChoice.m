function [CSA,CSWA,CSB,CSWB]=RandomChoice(A,WA,rA_thed,B,WB,rB_thed)
% num is the number of chosen points
% seqNo is the sequence array of the num chosen points
Num=size(A,2);
iniNo=randperm(Num);
CSA=A(:,iniNo(1:rA_thed));
disMat=distance(A,CSA);
[~,cluNo]=min(disMat,[],2);
CSWA=zeros(1,rA_thed);
for i=1:rA_thed
    ithNo=(cluNo==i);
    CSWA(i)=sum(WA(ithNo));
end

Num=size(B,2);
iniNo=randperm(Num);
CSB=B(:,iniNo(1:rB_thed));

disMat=distance(B,CSB);
[~,cluNo]=min(disMat,[],2);
CSWB=zeros(1,rB_thed);
for i=1:rB_thed
    ithNo=(cluNo==i);
    CSWB(i)=sum(WB(ithNo));
end

end