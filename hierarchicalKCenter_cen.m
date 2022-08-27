function [CSA,CSWA]=hierarchicalKCenter_cen(A,WA,num)
%sample points for kcenter+ 
Num=size(A,2);% the number of points in set A
iniNo=randi(Num);% the first number
cluNo=ones(1,Num);
disMat_all=distance(A,A);
disMat=disMat_all(iniNo,:);
cen=zeros(1,num);
cen(1)=iniNo;
for i=2:num
    [~,curNo]=max(disMat);
    cen(i)=curNo;
    tmpMat=disMat_all(curNo,:);
    [disMat,tmpSeq]=min([disMat;tmpMat],[],1);
    cluNo(tmpSeq==2)=i;
end
cen(cen==0)=[];
dim=size(A,1);
CSA=A(:,cen);
CSWA=zeros(1,num);
for i=1:num
    ithNo=(cluNo==i);
    CSWA(i)=sum(WA(ithNo));

end

















    
