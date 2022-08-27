clear
clc

load('en.mat');
load('es.mat');
load('es_en.mat');

m=size(es,1);
n=size(en,1);
indM=cell(m,1);
indN=cell(m,1);
for i=1:m
    indM{i}=(find(es(i)==es_en(:,1)));
end
for i=1:m
    sz=numel(indM{i});
    for j=1:sz
        for k=1:n
            temp=strfind(es_en(indM{i}(j),2),en(k));
            if numel(temp)>0
                indN{i}=[indN{i},k];
            end
        end
    end
end