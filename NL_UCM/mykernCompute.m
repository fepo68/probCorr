function k = mykernCompute(kern,x,x2)
% KERNCOMPUTE Compute the kernel given the parameters and x and x2.
% FORMAT
% DESC computes a kernel matrix for the given kernel type given an
% input data observations as columns.
% ARG kern : kernel structure to be computed.
% ARG x,x2 : input data matrix (rows are data features) to the kernel computation.
% RETURN K : computed elements of the kernel structure.

if nargin < 3
    x2 = x;
end

if strcmp(kern.type,'linear')
    
    k = x' * x2;
end
if strcmp(kern.type,'rbf')
    if ~isvector(x) && isvector(x2)
        [Md,K] = size(x);
        k = zeros(K,1);
        for i = 1:K
         k(i,1) = rbfKernCompute(kern, x(:,i)', x2');
        end

    else 
        k = rbfKernCompute(kern, x', x2');
    end
end


end