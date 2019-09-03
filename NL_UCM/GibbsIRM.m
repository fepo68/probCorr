function [partition_samples, partition_estimate, similarity,  West, partition_map, eta_map, logpost] = GibbsIRM(X, W, alpha, niter, type_network)

%
% GIBBSIRM performs Gibbs sampling for the infinite relational model
%   [partition_samples, partition_estimate, similarity,  West, partition_map, eta_map, logpost] = GibbsIRM(X, W, alpha, niter, type_network)
%
%--------------------------------------------------------------------------
% INPUTS
%   - X:    Sparse adjacency matrix
%   - W:    Sparse indicator matrix of missing links to be predicted
%   - alpha:Scale parameter of the IRM
%   - niter: Number of MCMC iterations
%   - type_network: 'Indirect' (default) or 'Directed'
%
% OUTPUTS
%   - partition_samples: Matrix of posterior partition samples
%   - partition_estimate: Bayesian point estimate of the partition
%   - similarity:  Posterior similarity matrix
%   - West: Posterior link probabilities
%   - partition_map: MAP estimate of the partition
%   - eta_map: MAP estimate of the between groups link probabilities
%   - logpost: logposterior at each iteration of the MCMC
%
% NOTE: The function is basically just a wrapper of the function
% IRMUnipartite for MCMC inference in IRM, by Morten Morup.
% 
% See IRMUNIPARTITE
%
%--------------------------------------------------------------------------




X = X - diag(diag(X));
W = W - diag(diag(W));
noc = 10;
opts.dZstep = 1;
opts.alpha = alpha;
if nargin<5
    opts.type = 'Undirected';    
else
    opts.type = type_network;
end
opts.nsampleiter = niter/2;  
opts.init_sample_iter=niter/2;
[logpost, cpu_time, Z, eta, sample, West, par] = IRMUnipartite(sparse(X),W,noc,opts);

%%
% Work on some of the outputs
for j=1:size(sample.MAP.Z, 2)
       partition_map(j) = find(sample.MAP.Z(:, j));
end
eta_map = sample.MAP.eta;

sample_Z = sample.Z;
nsamples = length(sample_Z);
n = size(X, 1);
partition_samples=zeros(n, nsamples);
for i=1:nsamples
   for j=1:size(sample_Z{i}, 2)
       partition_samples(j, i) = find(sample_Z{i}(:, j));
   end
end
% Get a Bayesian point estimate based on the posterior similarity matrix
% and Binder loss function
[partition_estimate, cost, similarity] = cluster_est_binder(partition_samples);
end
