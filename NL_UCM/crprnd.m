function [partition, m, K, partition_bin] = crprnd(alpha, n)

%
% CRPRND samples a partition from the Chinese restaurant process
%   [partition, m, K, partition_bin] = crprnd(alpha, n)
%
%--------------------------------------------------------------------------
% INPUTS
%   - alpha:    scale parameter of the CRP
%   - n:        number of objects in the partition
%
% OUTPUTS
%   - partition: Vector of length n. partition(i) is the cluster membership
%               of object i. Clusters are numbered by order of appearance
%   - m:        Vector of length n. m(j) is the size of cluster j.
%               m(j)=0 for j>K
%   - K:        Integer. Number of clusters
%   - partition_bin: locical matrix of size n*n. partition(i,j)=true if object
%               i is in cluster j, false otherwise
%--------------------------------------------------------------------------
% EXAMPLE
% alpha = 3; n= 100;
% [partition, m, K, partition_bin] = crprnd(alpha, n);
%--------------------------------------------------------------------------


% Check parameter
if alpha<=0
    error('Parameter alpha must be a positive scalar')
end

m = zeros(1, n);
partition = zeros(1, n);
partition_bin = false(n);

% Initialization
partition(1) = 1;
partition_bin(1,1) = true;
m(1) = 1;
K = 1;
% Iterations
for i=2:n
    % Compute the probability of joining an existing cluster or a new one
    proba = [m(1:K), alpha]/(alpha+i-1);
    % Sample from a discrete distribution w.p. proba
    u = rand;
    partition(i) = find(u<=cumsum(proba), 1);
    partition_bin(i,partition(i)) = true;
    % Increment the size of the cluster
    m(partition(i)) = m(partition(i)) + 1;
    % Increment the number of clusters if new
    K = K + isequal(partition(i), K+1);    
end