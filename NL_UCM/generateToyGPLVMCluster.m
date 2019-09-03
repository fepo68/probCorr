%% Genera Toy PhD. UCM-GPLVM, for our MultiView Learning Problem

clear all
close all
clc

K = 3;
V = 2;
Q = 3;
alpha = 0.1;
Dd = [3 3];
Nd = [100 100];
pathData = './toyData/';
% addpath(genpath('..\.\GPmat-master'));
addpath(genpath('..\.\MLtoolboxes'));
% rmpath('./private');
for D = 2:2 % for several domains experiments
    
%     Md = Md(D)*ones(1,D);
%     Nd = 200*ones(1,D);
    
    for i = 1:1
        
        %         seed = 1e5*3;
%         s = RandStream('mt19937ar','Seed',1e5*i);
%         RandStream.setGlobalStream(s);
%         [S,X,Fd] = createToyGPLVMCluster(Nd,Md,D,J,K); % For the nonlinear
        %         model
        [Sclust,Y,Fd,X] = createToyMultiviewWMMCluster(Nd,Dd,V,K,Q,alpha);
%         gscatter(X{1}(:,1),X{1}(:,2),S{1})
        
%         gsave([pathData,'synth',num2str(K),'exp',num2str(i),'D',num2str(D),'J',num2str(J),'GP09_11_17.mat'],'S','W','X','Nd','Md','D','J','K');
    end
    
end

 figure,plotMultiViewData(Y,Sclust);