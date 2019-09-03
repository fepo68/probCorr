%% Unsupervised Clustering Matching Via LVM
clear all
close all
clc
% Input: multiple domain datasets X,initial number of clusters J, hyperparameters a, b, r, alpha , iteration number T
% Output: cluster assignments S, projection matrices W
% s = RandStream('mt19937ar','Seed',1e5);
% RandStream.setGlobalStream(s);


% Kexp = 10; %for synth3

AR = zeros(1,5);
Rand =AR;
Mirkin = AR;
Hubert =AR;
ja = 1;
Clusters = {};
pathD ='./toyData/';
for i = 1:5
%     load synth5exp8  % synth3 J = 3;
    %     load synth3exp9 % synth3
%     load([pathD,'synth5exp2'])
%     load([pathD,'synth3exp1D2J5NLmap18oct18']);
    load  dataToyirisDataset1exp
    %     D = length(X);
    T = 300; % iterations
    %     J = 1; % initial number of clusters J
    alphaW = 0.1; % small variance for cprojection Matrices Wd
    K = 5; % dimensionality of the latent vector
    
    
    %         load(['synth',num2str(Kexp),'exp',num2str(i),'D',num2str(j)]);
    trueS = S;
    trueW = W;
    trueJ = J;
    clear J;
    s = RandStream('mt19937ar','Seed',1e5*i);
    RandStream.setGlobalStream(s);
    
    
    %% Linear model
    params.a = 1;
    params.b = 1;
    params.r = 1;
    params.gammaVal = 20;
    params.optimW = false;
%         %%%%% this to map W through basis functions
%         mapsToBF = false;
%         if mapsToBF == true
%             nfBasis = 50;
%             fBasis = 'sigmoid';
%             X = polyToyMapBasis(X,S,Md,Nd,nfBasis,fBasis,true);
%     
%         end
%     
    
    [X,S,W,params] = lvmInferCorrespondenceLin(T,D,K,alphaW,X,params);
    %% Nonlinear model
    %     [X,S,W,params] = lvmInferCorrespondenceNonLin(T,D,K,alphaW,X);
    
    %% Metrics
    Svect = [];
    Struevect = [];
    for d = 1:D
        Sd = S{d};
        Sdtrue = [trueS{d}]';
        Svect = [Svect;Sd];
        Struevect = [Struevect;Sdtrue];
        
    end
    [AR(i), Rand(i), Mirkin(i), Hubert(i)] =  valid_RandIndex(Svect,Struevect);
    Clusters{i,1} = Svect;
    Clusters{i,2} = Struevect;
    
end
