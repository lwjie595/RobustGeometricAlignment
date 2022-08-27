%%%%%sovling the Wasserstein distance for natural language

clear
clc
%% Bilinguistics
%load data
filefds=dir('Data/BiL/');
Num=numel(filefds);% the number of language pairs
transmatrxi=['es_en.mat','it_en.mat','ja2zh.mat','tr_en.mat','zh2en.mat'];
Amatrxi=['en.mat','en.mat','zh.mat','en.mat','en.mat'];
for i=3:3
    Name=filefds(i).name;% the name of file folder
    load(['Data/BiL/',Name,'/A.mat']);
    load(['Data/BiL/',Name,'/B.mat']);
    load(['Data/BiL/',Name,'/WA.mat']);
    load(['Data/BiL/',Name,'/WB.mat']);

    if Name=='zh-en'
        IT=load(['Data/BiL/',Name,'/W']);
        IT=IT';% the initialization of transformation matrix
    else
        IT=randn(size(A,1));
    end
   
      dim=size(A,1);
    n1=size(A,2);
    n2=size(B,2);

    tau=0;
csArr=0.02:0.02:0.1;% the array of core-set size
   runNo=length(csArr);
    csSz=1;% the number of core-sets
    %record result
    emdTab_cen=zeros(csSz,runNo);% record emd
    emdTab_means=zeros(csSz,runNo);
    emdTab_kcenter=zeros(csSz,runNo);
    emdTab_kmeans=zeros(csSz,runNo);
    emdTab_kmedian=zeros(csSz,runNo);
    emdTab_median=zeros(csSz,runNo);
    emdTab_RC=zeros(csSz,runNo);
    timTab_RC=zeros(csSz,runNo);
    timTab_kmedian=zeros(csSz,runNo);
    timTab_median=zeros(csSz,runNo);
    timTab_cen=zeros(csSz,runNo);% record time
    timeTab_means=zeros(csSz,runNo);
    timeTab_kcenter=zeros(csSz,runNo);
    timTab_kmeans=zeros(csSz,runNo);
    
    Error_kmedian=zeros(csSz,runNo);
    Error_median=zeros(csSz,runNo);
    Error_cen=zeros(csSz,runNo);% record time
    Error_means=zeros(csSz,runNo);
    Error_kcenter=zeros(csSz,runNo);
    Error_kmeans=zeros(csSz,runNo);
    Error_RC=zeros(csSz,runNo);
     Error_org=zeros(csSz,1);

timTab_org=zeros(csSz,1);
emdTab_org=zeros(csSz,1);


copA2=norCol(A);
copB2=norCol(B);
copWA2=WA/sum(WA);
copWB2=WB/sum(WB);
meanA=sum(bsxfun(@times,copWA2,copA2),2);
meanB=sum(bsxfun(@times,copWB2,copB2),2);
copA2=bsxfun(@minus,copA2,meanA);
copB2=bsxfun(@minus,copB2,meanB);




copA=norCol(A);
copB=norCol(B);
copWA=WA/sum(WA);
copWB=WB/sum(WB);
meanA=sum(bsxfun(@times,copWA,copA),2);
meanB=sum(bsxfun(@times,copWB,copB),2);
copA=bsxfun(@minus,copA,meanA);
copB=bsxfun(@minus,copB,meanB);
A=copA;
B=copB;

CoreNum=20;

  IT=eye(dim);


    tau_f=tau;

    tau_f2=tau_f;
   
 
%% Start algorithm
for outI=1:CoreNum
    dim=size(A,1);

    IT=randn(dim,dim);

        for inI=1:runNo   
            tau_outl=inI;
     rA_thed=ceil(n1*csArr(tau_outl));% the size of core-set of set A
    rB_thed=ceil(n2*csArr(tau_outl));

         %Original EMD
     disp('Original EMD');
    tic
    [~,~,TM]=SinkhornInit(A,B,WA,WB,IT,tau_f);
    [EMD1,FM_org]=Sinkhorn(TM*copA2,copB2,copWA2,copWB2,tau_f2);
    timTab_org(outI,inI)=toc;
    emdTab_org(outI,inI)=EMD1;
    
    % EMD based on Core-Set
    disp('EMD based on Core-Set');
       %Kcenter+
            tic
             [CSA,CSWA]=hierarchicalKCenter_cen(A,WA,rA_thed);
            [CSB,CSWB]=hierarchicalKCenter_cen(B,WB,rB_thed);
            [~,~,TM2]=SinkhornInit(CSA,CSB,CSWA,CSWB,IT,tau_f);
            [EMD,FM]=Sinkhorn(TM2*copA2,copB2,copWA2,copWB2,tau_f2);
            timTab_cen(outI,inI)=toc;
            emdTab_cen(outI,inI)=EMD;
            
               %Kcenter
            tic
            [CSA3,CSWA3]=KCenter(A,WA,rA_thed);
            [CSB3,CSWB3]=KCenter(B,WB,rB_thed);
            [~,~,TM2]=SinkhornInit(CSA3,CSB3,CSWA3,CSWB3,IT,tau_f);
            [EMD3,FM]=Sinkhorn(TM2*copA2,copB2,copWA2,copWB2,tau_f2);
            timeTab_kcenter(outI,inI)=toc;
            emdTab_kcenter(outI,inI)=EMD3;
            
               %Kmeans
             tic
            [CSA4,CSWA4]=kmeans_mm_weight(A,WA,rA_thed,10);
            [CSB4,CSWB4]=kmeans_mm_weight(B,WB,rB_thed,10);
            [~,~,TM2]=SinkhornInit(CSA4,CSB4,CSWA4,CSWB4,IT,tau_f);
            [EMD4,FM]=Sinkhorn(TM2*copA2,copB2,copWA2,copWB2,tau_f2);
            timTab_kmeans(outI,inI)=toc;
            emdTab_kmeans(outI,inI)=EMD4;
            
                 %StoasticOpt
        tic
            [~,~,TM2]=SinkhornInit_batch(A,B,WA,WB,IT,tau_f,csArr(tau_outl));
            [EMD4,FM]=Sinkhorn(TM2*copA2,copB2,copWA2,copWB2,tau_f2);
            timTab_median(outI,inI)=toc;
            emdTab_median(outI,inI)=EMD4;
            
            %Random
            tic
             [CSA2,CSWA2,CSB2,CSWB2]=hierarchicalKCenter_means(A,WA,rA_thed,B,WB,rB_thed);
            [~,~,TM2]=SinkhornInit(CSA2,CSB2,CSWA2,CSWB2,IT,tau_f);
            [EMD2,FM]=Sinkhorn(TM2*copA2,copB2,copWA2,copWB2,tau_f2);
            timeTab_means(outI,inI)=toc;
            emdTab_means(outI,inI)=EMD2;
            
            %Random+
            tic
            [CSA,CSWA,CSB,CSWB]=RandomChoice(A,WA,rA_thed,B,WB,rB_thed);
            [~,~,TM2]=SinkhornInit(CSA,CSB,CSWA,CSWB,IT,tau_f);
            [EMD,FM]=Sinkhorn(TM2*copA2,copB2,copWA2,copWB2,tau_f2);
            timTab_RC(outI,inI)=toc;
            emdTab_RC(outI,inI)=EMD;
            
            
        end
end
  
emdTab_cen=emdTab_cen';
emdTab_kcenter=emdTab_kcenter';
emdTab_means=emdTab_means';
    emdTab_kmeans=emdTab_kmeans';
    emdTab_RC= emdTab_RC';
    emdTab_org= emdTab_org';% record emd
    emdTab_median= emdTab_median';
    emdTab_kmedian= emdTab_kmedian';
    timTab_org= timTab_org';
    timTab_RC= timTab_RC';
    timTab_cen= timTab_cen';% record time
    timeTab_means= timeTab_means';
    timeTab_kcenter= timeTab_kcenter';
    timTab_kmeans= timTab_kmeans';  
    timTab_median= timTab_median';  
    timTab_kmedian= timTab_kmedian';  
    
    if ~exist(['./Result/BiL/nonoise/test/%s',filefds(i).name()],'dir')
        path{1}=['./Result/BiL/nonoise/test/',filefds(i).name()];
        mkdir (path{1})
    end
    save(sprintf('./Result/BiL/nonoise/test/%s/kcenter_sink_size%.2f.mat',filefds(i).name(),tau),'emdTab_org','timTab_org','emdTab_cen','timTab_cen','emdTab_means',...
    'timeTab_means','emdTab_kcenter','timeTab_kcenter','emdTab_kmeans','timTab_kmeans','emdTab_median','timTab_median',...
    'emdTab_kmedian','timTab_kmedian','timTab_RC','emdTab_RC','Error_RC','Error_kmedian','Error_kmeans','Error_kcenter','Error_means','Error_median',...
    'Error_cen');
end

datestr(now)