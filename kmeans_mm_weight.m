function [centers_,weight_sum]=kmeans_mm_weight(X,weight,k,max_iters,centers_initial)

X=X';
weight=weight';
weight_sum=zeros(size(k,1),1);
if nargin==5
    centers=centers_initial;
else

centers_index=plusintinal(X,k);
centers=X(centers_index,:);
end



[distance,nst_ct]=min(pdist2(X,centers,'squaredeuclidean'),[],2);

cost=sum(distance.*weight);
for i=1:k
    centers_(i,:)=update_centers(X,weight,i,nst_ct);
    weight_sum(i,:)=sum(weight(nst_ct==i,:));
end

[distance_,nst_ct]=min(pdist2(X,centers_,'squaredeuclidean'),[],2);

cost_=sum(distance_.*weight);
num=0;
while abs(cost-cost_)>1e-6
if num>=max_iters
break
end

centers=centers_;
cost=cost_;

for i=1:k
    centers_(i,:)=update_centers(X,weight,i,nst_ct);
    weight_sum(i,:)=sum(weight(nst_ct==i,:));
end
[distance_,nst_ct]=min(pdist2(X,centers_,'squaredeuclidean'),[],2);

cost_=sum(distance_.*weight);



num=num+1;


end
centers_=centers_';
weight_sum=weight_sum';
end


function centers_i=update_centers(X,weight_rm_out,i,nst_ct)
X=X(nst_ct==i,:);
weight=weight_rm_out(nst_ct==i);
X=X.*weight/sum(weight);
centers_i=sum(X,1);

end



function InitPoints=plusintinal(X,k)
    n=size(X,1);
    InitPoints=randi(n,1);
    disMat_all=distance(X',X');
    disMat_all(disMat_all<0)=0;
disMat=disMat_all(:,InitPoints);

     for Num=2:k
             chosenrange=[1:n];

            Prob=double(sqrt(disMat)/sum(sqrt(disMat)))';
            Prob=Prob/sum(Prob);
             curNo=randsrc(1,1,[chosenrange;Prob]);

               tmpMat=disMat_all(:,curNo);
            [disMat,tmpSeq]=min([disMat tmpMat],[],2);
            InitPoints=[InitPoints, curNo];
     end
end




