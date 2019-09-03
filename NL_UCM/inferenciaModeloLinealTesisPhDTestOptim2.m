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
for i = 1:1
    load synth3exp1D2 % synth3
    % load dataToyIwatamodel31Ene162D40Ndj5J5K % synth5
    % load dataToyIwatamodel31Ene162D40Ndj5J10K % synth10
    
    %     D = length(X);
    T = 100; % iterations
    J = 1; % initial number of clusters J
    alphaW = 0.05; % small variance for projection Matrices Wd
    K = 5; % dimensionality of the latent vector
    
    
    
    
    
    %         load(['synth',num2str(Kexp),'exp',num2str(i),'D',num2str(j)]);
    trueS = S;
    trueW = W;
    trueJ = J;
    clear J S
    s = RandStream('mt19937ar','Seed',1e5*i);
    RandStream.setGlobalStream(s);
    
    
    %% Linear model
%     [X,S,W,params] = lvmInferCorrespondenceLin(T,D,K,alphaW,X);
    %% Nonlinear model
    [X,Sout,W,params] = lvmInferCorrespondenceNonLin(T,D,K,alphaW,X);
    
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
