%% Unsupervised Clustering Matching Via LVM
clear
close all
clc
% Input: multiple domain datasets X,initial number of clusters J, hyperparameters a, b, r, alpha , iteration number T
% Output: cluster assignments S, projection matrices W
% s = RandStream('mt19937ar','Seed',1e5);
% RandStream.setGlobalStream(s);


Kexp = 5; %for synth3
D = 2;
AR = zeros(10,5);
Rand =AR;
Mirkin = AR;
Hubert =AR;
pathD = './toyData/';
for j = 1:1
    %
    %     load([pathD,'synth',num2str(Kexp),'exp',num2str(j),'D',num2str(D),'17oct17.mat'])
    %     load([pathD,'synth3exp',num2str(j),'D2J5NLmap18oct18.mat'])
    fprintf('processing dataset number %d ...\n',j);
    %     load(['dataToyirisDataset',num2str(j),'exp'])
            load(['dataToywineDataset',num2str(j),'exp'])
%     load([pathD,'dataToyMNISTDataset',num2str(j),'exp'])
%     load(['dataToyglassDataset',num2str(j),'exp'])
    %         load([pathD,'synth3exp1D2J5NLmap23oct17']);
    %     load dataToyIwatamodel18Ene162D50Ndj2J3K
    %     load dataToyIwatamodel03Feb16_12D40Ndj5J5K % synth3
    % load dataToyIwatamodel31Ene162D40Ndj5J5K % synth5
    % load dataToyIwatamodel31Ene162D40Ndj5J10K % synth10
    
    %% this to map X through basis functions
    s = RandStream('mt19937ar','Seed',1e5);
    RandStream.setGlobalStream(s);
    mapsToBF = true;
    if mapsToBF == true
        %         kvar = 0.01;
        %         bias = 0.05;
        %         nfBasis = 4;
        %         fBasis = 'sigmoid';
        %         %         X = polyToyMapBasis(X,S,Md,Nd,nfBasis,fBasis,true,kvar,bias);
        Nrbf = 5;
        X = randFeatureExpansionCompute(X,Nd,Md,Nrbf,true,2);
        
    end
    
    %% model parameters
    D = length(X);
    T = 100; % iterations
    J = 1; % initial number of clusters J
    alphaW = 0.1; % small variance for projection Matrices Wd
    K = 3; % dimensionality of the latent vector
    
    trueS = S;
%     trueW = W;
    trueJ = 2;
    clear J;
    
    for i = 2:2
        
        s = RandStream('mt19937ar','Seed',1e5*i);
        RandStream.setGlobalStream(s);
        
        params.a = 1;
        params.b = 1;
        params.r = 1;
        params.gammaVal = 2;
        params.optimW = true;
        [Xout,Sout,W,params] = lvmInferCorrespondenceLin(T,D,K,alphaW,X,params);
        
        %% Metrics
        Svect = [];
        Struevect = [];
        for d = 1:D
            Sd = Sout{d};
            Sdtrue = [trueS{d}]';
            Svect = [Svect;Sd];
            Struevect = [Struevect;Sdtrue];
            
        end
        [AR(j,i), Rand(j,i), Mirkin(j,i), Hubert(j,i)] =  valid_RandIndex(Svect,Struevect);
        
    end
end
