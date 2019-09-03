% Clustering Using Gaussian Mixture Distributions
% Gaussian mixture distributions can be used for clustering data, by realizing that the multivariate normal components of the fitted model can represent clusters.
clear all
close all
clc
% To demonstrate the process, first generate some simulated data from a mixture of two bivariate Gaussian 
% distributions using the mvnrnd function.
rng default;  % For reproducibility
mu1 = [1 2];
sigma1 = [3 .2; .2 2];
mu2 = [-1 -2];
sigma2 = [2 0; 0 1];
X = [mvnrnd(mu1,sigma1,200);mvnrnd(mu2,sigma2,100)];

load LventricleSIHKS
y1 =[shape{1}.X';shape{1}.Y';shape{1}.Z';shape{1}.sihks(:,1)'];
y2 =[shape{2}.X';shape{2}.Y';shape{2}.Z';shape{2}.sihks(:,2)'];
y = [y1 y2];

X = y';
kCluster = 20;


scatter(X(:,1),X(:,2),10,'ko')

% Fit a two-component Gaussian mixture distribution. Here, you know the correct number of components to use. 
% In practice, with real data, this decision would require comparing models with different numbers of components.
options = statset('Display','final');
gm = fitgmdist(X,kCluster,'Options',options);
% 33 iterations, log-likelihood = -1210.59
% Plot the estimated probability density contours for the two-component mixture distribution.
% The two bivariate normal components overlap, but their peaks are distinct. 
% This suggests that the data could reasonably be divided into two clusters.
% hold on
% ezcontour(@(x,y)pdf(gm,[x y]),[-8 6],[-8 6]);
% hold off

% Partition the data into clusters using the cluster method for the fitted mixture distribution. 
% The cluster method assigns each point to one of the two components in the mixture distribution.
idx = cluster(gm,X);
cluster1 = (idx == 1);
cluster2 = (idx == 2);

scatter(X(cluster1,1),X(cluster1,2),10,'r+');
hold on
scatter(X(cluster2,1),X(cluster2,2),10,'bo');
hold off
legend('Cluster 1','Cluster 2','Location','NW')

% Each cluster corresponds to one of the bivariate normal components in the mixture distribution. 
% cluster assigns points to clusters based on the estimated posterior probability that a point came from a
% component; each point is assigned to the cluster corresponding to the highest posterior probability.
% The posterior method returns those posterior probabilities. 
% For example, plot the posterior probability of the first component for each point.
P = posterior(gm,X);

scatter(X(cluster1,1),X(cluster1,2),10,P(cluster1,1),'+')
hold on
scatter(X(cluster2,1),X(cluster2,2),10,P(cluster2,1),'o')
hold off
legend('Cluster 1','Cluster 2','Location','NW')
clrmap = jet(80); colormap(clrmap(9:72,:))
ylabel(colorbar,'Component 1 Posterior Probability')