%% Unsupervised Clustering Matching Via GPLVM
clear
close all
clc
% Input: multiple domain datasets X,initial number of clusters J, hyperparameters a, b, r, alpha , iteration number T
% Output: cluster assignments S, projection matrices W
% s = RandStream('mt19937ar','Seed',1e5);
% RandStream.setGlobalStream(s);
addpath(genpath('E:\Clases Doctorado\Applied Bayesian Nonparametrics\MLtoolboxes\'));


D = 2;
AR = zeros(10,5);
Rand =AR;
Mirkin = AR;
Hubert =AR;
pathD = './toyData/';
for j = 1:5
    
    addpath(genpath('E:\Clases Doctorado\Applied Bayesian Nonparametrics\GPmat-master'));
    fprintf('processing dataset number %d ...\n',j);
%             load(['dataToyirisDataset',num2str(j),'exp'])
%                 load(['dataToywineDataset',num2str(j),'exp'])
        load([pathD,'dataToyMNISTDataset',num2str(j),'exp'])
%     load dataToyMNISTDataset1exp
    % %     load(['dataToyglassDataset',num2str(j),'exp'])
    
    %% this to compute object data X through a GPLVM
    
    mapsGP = true;
    flagN = false;
    if mapsGP == true
        Yall = X; % input data "observed space"
        %% GPLVM + UCM
        Xgplvm = {};
         s = RandStream('mt19937ar','Seed',1e5*j);
            RandStream.setGlobalStream(s);
        for v = 1:D
           
            Y = Yall{v};
            % First we represent our objects by using a GPLVM per domain
            
            % Set up model
            options = fgplvmOptions('pitc');
            options.optimiser = 'scg';
            d = size(Y, 2);
            %             latentDim = round(d/5);
            latentDim = 5;
            
            model = fgplvmCreate(latentDim, d, Y, options);
            
            % Optimise the model.
            iters = 100;
            display = 1;
            
            model = fgplvmOptimise(model, display, iters);
            if flagN == true
                Xgplvm{v} = zscore(model.X);
            else
                Xgplvm{v} = model.X;
            end
            clear model
        end
        
        
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
    X = Xgplvm;
    rmpath(genpath('E:\Clases Doctorado\Applied Bayesian Nonparametrics\GPmat-master'));
    for i = 1:5
        
        s = RandStream('mt19937ar','Seed',1e5*i);
        RandStream.setGlobalStream(s);
        
        params.a = 1;
        params.b = 1;
        params.r = 1;
        params.gammaVal = 2;
        params.optimW = true;
        iGMM = false;
        if iGMM == true
            [Xout,Sout,W,params] = lvmInferCorrespondenceLin(T,D,K,alphaW,X,params);
        else
            Data = [];
            for d = 1:D
                Data = [Data;X{d}];
            end
            K= 3;
            [Px, modelG] = gmmSingleView(Data, K);
            Sout = {};
            id=[1,201];
            for d = 1:D
                Pxd = Px(id(d):id(d)+199,:);
                [aa,pos] = max(Pxd');
                Sout{d} = pos';
            end
            clear modelG
        end
        
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








