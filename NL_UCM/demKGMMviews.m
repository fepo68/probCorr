%% Kernel Trick Embedded for GMM
clear all
close all
clc

addpath(genpath('E:\Clases Doctorado\Tesis Doctoral\Correspondence Problem\IWATA\warped-mixtures-master\warped-mixtures-master'));
addpath(genpath('E:\Clases Doctorado\Applied Bayesian Nonparametrics\GPmat-master\'));
addpath(genpath('E:\Clases Doctorado\Applied Bayesian Nonparametrics\MLtoolboxes\'));
% load streamOKcircle.mat
% s = RandStream('mt19937ar','Seed',1e5);
% RandStream.setGlobalStream(s);


N = 200;
K = 2;
Ko = 2;
T = 100;
% load spiral2
% load pinwheel_N50K5
% [ X , y] = gen_multi_spiral_data( N, K );
% [ X , y ] = gen_circles_data( N, K );
% load('./data/circles_N200K2');
load('./data/circles_N100K2');
% load('./toydataClust/data_TwoDiamonds.mat')
% X = D;
% y = L;
[N,D] = size(X);
X = zscore(X);
% ./repmat(1.5*max(X),N,1);
CMo = jet(Ko); % Set colors for clusters
CM = jet(K); % Set colors for clusters
figure,
subplot(1,3,1),plot(X(y==1,1),X(y==1,2),'o','color',CMo(1,:),'MarkerSize',5,...
    'MarkerFaceColor', CMo(1,:)),title('Real Clusters');
hold on
forms = {'b.','k.','g.','m.','y.'};
for k = 2:Ko
plot(X(y==k,1),X(y==k,2),'o','color',CMo(k,:),'MarkerSize',5,...
    'MarkerFaceColor', CMo(k,:));
end
[pli,Ghat] = myKernelGMMCompute(X,K,T);
% [pli,Ghat] = myKernelGMMComputeVJieYu2012(X,K,T);
% [~ , idx] = find(pli>=0.6);
[~,idx] = max(pli');
[ARk, Randk, Mirkink, Hubertk] =  valid_RandIndex(idx,y);
% figure,
subplot(1,3,2),plot(X(idx==1,1),X(idx==1,2),'o','color',CM(1,:),'MarkerSize',5,...
    'MarkerFaceColor', CM(1,:)),title('Kernel-GMM')
hold on
for k = 2:length(unique(idx))
plot(X(idx==k,1),X(idx==k,2),'o','color',CM(k,:),'MarkerSize',5,...
    'MarkerFaceColor', CM(k,:));
end



%% GMM Part

[Px, model] = gmmSingleView(X, K);
[~ , idx1] = max(Px');
[AR, Rand, Mirkin, Hubert] =  valid_RandIndex(idx1,y);
% figure,
subplot(1,3,3),plot(X(idx1==1,1),X(idx1==1,2),'o','color',CM(1,:),'MarkerSize',5,...
    'MarkerFaceColor', CM(1,:)),title('GMM')
hold on
for k = 2:K
plot(X(idx1==k,1),X(idx1==k,2),'o','color',CM(k,:),'MarkerSize',5,...
    'MarkerFaceColor', CM(k,:))
end