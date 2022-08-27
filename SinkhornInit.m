function [EMD,FM,TM]=SinkhornInit(A,B,WA, WB,TM,tau)
% Data Matrices: A, B
% Weights (Disributions): WA, WB
% TM: the initialization of rotation matrix
M=size(A,2);% the number of points in set A
N=size(B,2);% the number of points in set B
% Normalize each column vector in matrices A and B
WA=WA(:,sum(A,1)~=0);
WB=WB(:,sum(B,1)~=0);
A=A(:,sum(A,1)~=0);
B=B(:,sum(B,1)~=0);


% Make two weight vectors be distributions
WA=WA/sum(WA);
WB=WB/sum(WB);
%% Transition Vector

%% Threshold
itThred=1e-6;
%% Iteration Algorithm

dim=size(A,1);% the dimensionality of each point
[U,~,V]=svd(TM);
TM=U*V';
TM_A=TM*A;

Thre=-1;
num=0;
[EMD,FM]=Sinkhorn(TM_A,B,WA,WB,tau);
while 1
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

    resM=double(B*FM'*A');
    [U,~,V]=svd(resM);


    TM=U*V';
    TM_A=TM*A;
    num=num+1;
    if num>50
        break;
    end
end