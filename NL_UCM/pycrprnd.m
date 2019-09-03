function [partition, m, K, partition_bin] = pycrprnd(alpha, sigma, n)

%
% PYCRPRND samples a partition from the Pitman-Yor Chinese restaurant process
%   [partition, m, K, partition_bin] = pycrprnd(alpha, sigma, n)
%
%--------------------------------------------------------------------------
% INPUTS
%   - alpha:    scale parameter of the PY CRP
%   - sigma:    discount parameter of the PY CRP
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


m = zeros(n, 1);
partition = zeros(n, 1);
partition_bin = false(n);

% Initialization
partition(1) = 1;
partition_bin(1,1) = true;
m(1) = 1;
K = 1;
% Iterations
for i=2:n
    % Compute the probability of joining an existing cluster or a new one
    proba = [m(1:K)-sigma; alpha + sigma*K]/(alpha+i-1);
    % Sample from a discrete distribution w.p. proba
    u = rand;
    partition(i) = find(u<=cumsum(proba), 1);
    partition_bin(i,partition(i)) = true;
    % Increment the size of the cluster
    m(partition(i)) = m(partition(i)) + 1;
    % Increment the number of clusters if new
    K = K + isequal(partition(i), K+1);    
end
partition_bin = partition_bin(:, 1:K);