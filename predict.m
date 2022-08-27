function pred=predict(TM,A,B,label1,label2,FM)
    
    TM_A=TM*A;
    Dis2=distance(TM_A,B);
    nB=size(label2,2);
    [~,closepoint2]=min(Dis2,[],1);
    label_predict2=label1(closepoint2');
    count1=sum(label_predict2-label2<1e-4);
    pred=count1/nB;


    

end