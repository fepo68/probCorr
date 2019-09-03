%% Unsupervised Clustering Matching Via LVM
clear
close all
clc
% Input: multiple domain datasets X,initial number of clusters J, hyperparameters a, b, r, alpha , iteration number T
% Output: cluster assignments S, projection matrices W
% s = RandStream('mt19937ar','Seed',1e5);
% RandStream.setGlobalStream(s);


Kexp = 3; %for synth3

AR = zeros(10,5);
Rand =AR;
Mirkin = AR;
Hubert =AR;
for j = 1:1
    
    
    
    
    
    load(['toyData/synth',num2str(Kexp),'exp',num2str(j),'D226feb18']);
    % load dataToyIwatamodel18Ene162D50Ndj2J3K
    %     load dataToyIwatamodel03Feb16_12D40Ndj5J5K % synth3
    % load dataToyIwatamodel31Ene162D40Ndj5J5K % synth5
    % load dataToyIwatamodel31Ene162D40Ndj5J10K % synth10
    
    D = length(X);
    T = 100; % iterations
    J = 1; % initial number of clusters J
    alphaW = 0.1; % small variance for projection Matrices Wd
    K = 5; % dimensionality of the latent vector
    
    trueS = S;
    trueW = W;
    trueJ = J;
    clear J;
    
    for i = 1:1
        
        s = RandStream('mt19937ar','Seed',1e5*i);
        RandStream.setGlobalStream(s);
        
        params.a = 1;
        params.b = 1;
        params.r = 1;
        params.gammaVal = 1;
        params.optimW = true;
        [X,S,W,params] = lvmInferCorrespondenceLin(T,D,K,alphaW,X,params);
        
        %% Metrics
        Svect = [];
        Struevect = [];
        for d = 1:D
            Sd = S{d};
            Sdtrue = [trueS{d}]';
            Svect = [Svect;Sd];
            Struevect = [Struevect;Sdtrue];
            
        end
        [AR(j,i), Rand(j,i), Mirkin(j,i), Hubert(j,i)] =  valid_RandIndex(Svect,Struevect);
        
    end
end
