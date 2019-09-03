%% Databases to UCM-LVM model
clear all
close all
clc

% load  dataToyIwatamodel27Ene162D40Ndj5J10K
for rep = 1:5
    s = RandStream('mt19937ar','Seed',1e5*rep);
    RandStream.setGlobalStream(s);
    
    dataSets = {'iris','wine','glass'};
    dataSet = dataSets{2};
    if strcmp(dataSet,'iris')
        load iris_dataset
        D = 2;
        Nd = [150,150];
        Md = [2,2];
        idPerm = randperm(4);
        X{1} = irisInputs(idPerm(1:2),:)';
        X{2} = irisInputs(idPerm(3:4),:)';
        J = 3;
        K = 3;
        W{1} = eye(Md(1),K);
        W{2} = eye(Md(2),K);
        
        S{1} = [ones(1,50),2*ones(1,50),3*ones(1,50)];
        S{2} = [ones(1,50),2*ones(1,50),3*ones(1,50)];
%     end
    elseif strcmp(dataSet,'wine')
        load wine_dataset
        [targets,~] = find(wineTargets==1);
        S{1} = targets';
        S{2} = targets';
        J = length(unique(targets));
        id1 = randperm(13);
        idDa = id1(1:round(length(id1)*.4));
        idDb = id1(round(length(id1)*.4)+1:end);
        X{1} = wineInputs(idDa,:)';
        X{2} = wineInputs(idDb,:)';
        D = 2;
        Nd = [178,178];
        Md = [length(idDa),length(idDb)];
        
        K = 3;
        W{1} = eye(Md(1),K);
        W{2} = eye(Md(2),K);
    elseif strcmp(dataSet,'glass')   
        load glass_dataset
        [targets,~] = find(glassTargets==1);
        S{1} = targets';
        S{2} = targets';
        J = length(unique(targets));
        id1 = randperm(9);
        idDa = id1(1:round(length(id1)*.5));
        idDb = id1(round(length(id1)*.5)+1:end);
        
        X{1} = glassInputs(idDa,:)';
        X{2} = glassInputs(idDb,:)';
        D = 2;
        Nd = [214,214];
        Md = [length(idDa),length(idDb)];
        
        K = 3;
        W{1} = eye(Md(1),K);
        W{2} = eye(Md(2),K);
    end
    save(['dataToy',dataSet,'Dataset',num2str(rep),'exp']);
end