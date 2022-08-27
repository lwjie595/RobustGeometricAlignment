function [CSA,CSWA,CSB,CSWB]=hierarchicalKCenter_means(A,WA,rA_thed,B,WB,rB_thed)
%Sample points for kmeans
Num=size(A,2);
iniNo=randperm(Num);
CSA=A(:,iniNo(1:rA_thed));

CSWA=zeros(1,rA_thed);
for i=1:rA_thed

    CSWA(i)=sum(WA,'all')/rA_thed;
end

Num=size(B,2);
iniNo=randperm(Num);
CSB=B(:,iniNo(1:rB_thed));

CSWB=zeros(1,rB_thed);
for i=1:rB_thed

    CSWB(i)=sum(WB,'all')/rB_thed;
end

end








% 
