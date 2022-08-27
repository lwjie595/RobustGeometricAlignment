function DA_caltech_mulprocess()
%%%%%sovling the Wasserstein distance for domain adaption

clear
clc
%load data
matname={'amazon','caltech10','dslr','webcam'};
for i=3:4


for Bmat=1:4
     disp('i Bmat')
  i,Bmat
    load(sprintf('Data/DA/extend/amazon_small/%s.mat',matname{i}));

A=fts';
label1=labels;
    if Bmat==i
        continue;
    end
    load(sprintf('Data/DA/extend/amazon_small/%s.mat',matname{Bmat}));
    B=fts';
    label2=labels;
     dim=size(A,1);
    n1=size(A,2);
    n2=size(B,2);
    WA=ones(1,n1);
    WB=ones(1,n2);

tau=0;

csArr=0.02:0.02:0.1;% the array of core-set size
%record result
    csSz=1;% the number of core-sets
    runNo=length(csArr);
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
outI=1;

timTab_org=zeros(csSz,1);
emdTab_org=zeros(csSz,1);

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

tau_f=tau;

dim=size(A,1);

IT=eye(dim);


%% Start algorithm
for outI=1:CoreNum


IT=eye(dim)+0.1*randn(dim,dim);
  


       
        for inI=1:runNo
            tau_outl=inI;
              rA_thed=ceil(n1*csArr(tau_outl));% the size of core-set of set A
    rB_thed=ceil(n2*csArr(tau_outl));% the size of core-set of set B
% Original
    disp('Original EMD');
    tic
    [EMD1,FM_org,TM]=SinkhornInit(A,B,WA,WB,IT,tau_f);
    timTab_org(outI,inI)=toc;
    emdTab_org(outI,inI)=EMD1;
    Error_org(outI,inI)=predict(TM,copA,copB,label1,label2,FM_org);

       disp('EMD based on Core-Set');
         %Kcenter+
            tic
            [CSA,CSWA]=hierarchicalKCenter_cen(A,WA,rA_thed);
            [CSB,CSWB]=hierarchicalKCenter_cen(B,WB,rB_thed);
            [~,~,TM2]=SinkhornInit(CSA,CSB,CSWA,CSWB,IT,tau_f);
            [EMD,FM]=Sinkhorn(TM2*copA,copB,copWA,copWB,tau_f);
            timTab_cen(outI,inI)=toc;
            emdTab_cen(outI,inI)=EMD;
            Error_cen(outI,inI)=predict(TM2,copA,copB,label1,label2,FM);
        
          
                %Kcenter
            tic
            [CSA3,CSWA3]=KCenter(A,WA,rA_thed);
            [CSB3,CSWB3]=KCenter(B,WB,rB_thed);
            [~,~,TM2]=SinkhornInit(CSA3,CSB3,CSWA3,CSWB3,IT,tau_f);
            
            [EMD3,FM]=Sinkhorn(TM2*copA,copB,copWA,copWB,tau_f);
            timeTab_kcenter(outI,inI)=toc;
            emdTab_kcenter(outI,inI)=EMD3;
            Error_kcenter(outI,inI)=predict(TM2,copA,copB,label1,label2,FM);
             %Kmeans 
             tic
            [CSA4,CSWA4]=kmeans_mm_weight(A,WA,rA_thed,10);
            [CSB4,CSWB4]=kmeans_mm_weight(B,WB,rB_thed,10);
            [~,~,TM2]=SinkhornInit(CSA4,CSB4,CSWA4,CSWB4,IT,tau_f);            
            [EMD4,FM]=Sinkhorn(TM2*copA,copB,copWA,copWB,tau_f);
            timTab_kmeans(outI,inI)=toc;
            emdTab_kmeans(outI,inI)=EMD4;
            Error_kmeans(outI,inI)=predict(TM2,copA,copB,label1,label2,FM);

             %StoasticOpt
               tic
            [~,~,TM2]=SinkhornInit_batch(A,B,WA,WB,IT,tau_f,csArr(tau_outl));
            [EMD4,FM]=Sinkhorn(TM2*copA,copB,copWA,copWB,tau_f);
            timTab_median(outI,inI)=toc;
            emdTab_median(outI,inI)=EMD4;
            Error_median(outI,inI)=predict(TM2,copA,copB,label1,label2,FM);
            
              %Random
            tic
             [CSA2,CSWA2,CSB2,CSWB2]=hierarchicalKCenter_means(A,WA,rA_thed,B,WB,rB_thed);
            [~,~,TM2]=SinkhornInit(CSA2,CSB2,CSWA2,CSWB2,IT,tau_f);
            [EMD2,FM]=Sinkhorn(TM2*copA,copB,copWA,copWB,tau_f);        
            timeTab_means(outI,inI)=toc;
            emdTab_means(outI,inI)=EMD2;
            Error_means(outI,inI)=predict(TM2,copA,copB,label1,label2,FM);
            
            %Random+
            tic
            [CSA,CSWA,CSB,CSWB]=RandomChoice(A,WA,rA_thed,B,WB,rB_thed);
            [~,~,TM2]=SinkhornInit(CSA,CSB,CSWA,CSWB,IT,tau_f);
            
            [EMD,FM]=Sinkhorn(TM2*copA,copB,copWA,copWB,tau_f);
            timTab_RC(outI,inI)=toc;
            emdTab_RC(outI,inI)=EMD;
            Error_RC(outI,inI)=predict(TM2,copA,copB,label1,label2,FM);
            
            
        end
end

emdTab_cen=emdTab_cen';
emdTab_kcenter=emdTab_kcenter';
emdTab_means=emdTab_means';
    emdTab_kmeans=emdTab_kmeans';
    emdTab_RC= emdTab_RC';
    emdTab_org= emdTab_org';
    emdTab_median= emdTab_median';
    emdTab_kmedian= emdTab_kmedian';
    timTab_org= timTab_org';
    timTab_RC= timTab_RC';
    timTab_cen= timTab_cen';
    timeTab_means= timeTab_means';
    timeTab_kcenter= timeTab_kcenter';
    timTab_kmeans= timTab_kmeans';  
    timTab_median= timTab_median';  
    timTab_kmedian= timTab_kmedian';  
    if ~exist(['./Result/DA/nonoise/test/%s',matname{i}],'dir')
            path{1}=['./Result/DA/nonoise/test/',matname{i}];
            mkdir (path{1})
        end
        save(sprintf('./Result/DA/nonoise/test/%s/%s_kcenter_sink_size%.2f.mat',matname{i},matname{Bmat},tau),'emdTab_org','timTab_org','emdTab_cen','timTab_cen','emdTab_means',...
        'timeTab_means','emdTab_kcenter','timeTab_kcenter','emdTab_kmeans','timTab_kmeans','emdTab_median','timTab_median',...
        'emdTab_kmedian','timTab_kmedian','timTab_RC','emdTab_RC','Error_RC','Error_kmedian','Error_kmeans','Error_kcenter','Error_means','Error_median',...
        'Error_cen','Error_org');
        end
end
datestr(now)
