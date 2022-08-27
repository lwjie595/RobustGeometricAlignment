function [EMD,FM,TM]=SinkhornInit_batch(A,B,WA, WB,TM,tau,batch)
% Data Matrices: A, B
% Weights (Disributions): WA, WB
% TM: the initialization of rotation matrix
M=size(A,2);% the number of points in set A
N=size(B,2);% the number of points in set B

WA_org=WA(:,sum(A,1)~=0);
WB_org=WB(:,sum(B,1)~=0);
A_org=A(:,sum(A,1)~=0);
B_org=B(:,sum(B,1)~=0);


WA_org=WA_org/sum(WA_org);
WB_org=WB_org/sum(WB_org);
%% Transition Vector

%% Threshold
alpha=1e-1;
itThred=1e-6;
%% Iteration Algorithm

sample_A=randsample(M,ceil(M*batch));
sample_B=randsample(N,ceil(N*batch));
A=A_org(:,sample_A);
B=B_org(:,sample_B);
WA=WA_org(:,sample_A);
WB=WB_org(:,sample_B);
WA=WA/sum(WA);
WB=WB/sum(WB);

dim=size(A,1);% the dimensionality of each point
[U,~,V]=svd(TM);
TM=U*V';


TM_A=TM*A;
Thre=-1;
num=0;


[EMD,FM]=Sinkhorn(TM_A,B,WA,WB,tau);
while 1
    % Solving Eq. (9)
    % EMD is the earth move distance; FM is the flow matrix
    
    FM1=FM;
    EMD1=EMD;
    [EMD,FM]=Sinkhorn(TM_A,B,WA,WB,tau);
    if abs(Thre-EMD)<=itThred*EMD
        break;
    end
    if sum(isnan(FM))
        EMD=EMD1;    
        FM=FM1;
        break;
    end
    Thre=EMD;
    % Solving Eq. (10)

    resM=double(B*FM'*A');
    [U,~,V]=svd(resM);
    TM=U*V';

   
    sample_A=randsample(M,ceil(M*batch));
    sample_B=randsample(N,ceil(N*batch));
    A=A_org(:,sample_A);
    B=B_org(:,sample_B);
    WA=WA_org(:,sample_A);
    WB=WB_org(:,sample_B);
    WA=WA/sum(WA);
    WB=WB/sum(WB);
    
    TM_A=TM*A;
    num=num+1;
    if num>50
        break;
    end
end
