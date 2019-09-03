clear all
close all
clc

addpath(genpath('E:\Clases Doctorado\Applied Bayesian Nonparametrics\MLtoolboxes\'));
addpath(genpath('E:\Clases Doctorado\Applied Bayesian Nonparametrics\GPmat-master'));
addpath(genpath('E:\Clases Doctorado\Applied Bayesian Nonparametrics\Shared GPLVM\SGPLVM-master\matlab'));
D = 2;
AR = zeros(10,5);
Rand =AR;
Mirkin = AR;
Hubert =AR;
pathD = './toyData/';
for j = 1:5
    
    
    fprintf('processing dataset number %d ...\n',j);
    %             load(['dataToyirisDataset',num2str(j),'exp'])
    %                 load(['dataToywineDataset',num2str(j),'exp'])
    load([pathD,'dataToyMNISTDataset',num2str(j),'exp'])
    
    % Y_train = X{1};
    % Z_train = X{2};
    for i = 1:5
        s = RandStream('mt19937ar','Seed',1e5*i);
        RandStream.setGlobalStream(s);
        
        Y_train = zscore(X{1});
        Z_train = zscore(X{2});
        
        sX = sharedGPLVM(Y_train,Z_train);
        %         rmpath(genpath('E:\Clases Doctorado\Applied Bayesian Nonparametrics\GPmat-master'));
        Data = [sX.X(:,2:3);sX.X(:,4:5)];
        K= 10;
        [Px, modelG] = gmmSingleView(Data, K);
        [~,idx] = max(Px');
        y = [S{1}';S{2}'];
        [AR(j,i), Randk, Mirkink, Hubertk] =  valid_RandIndex(idx,y);
        clear sX Px
    end
    %     rmpath(genpath('E:\Clases Doctorado\Applied Bayesian Nonparametrics\MLtoolboxes\'));
    %     rmpath(genpath('E:\Clases Doctorado\Applied Bayesian Nonparametrics\GPmat-master'));
    %     rmpath(genpath('E:\Clases Doctorado\Applied Bayesian Nonparametrics\Shared GPLVM\SGPLVM-master\matlab'));
    
end