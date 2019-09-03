function [out, x0, y0, weights] = graphonirmrnd(alpha, sigma, beta, K)

weights = pystickbreaking(alpha, sigma, K);
cumweights = cumsum(weights);
% cumweights(end)>1
% sum(weights)>1
% pause
eta = betarnd(beta, beta, K, K);
stepsize = .005;
x0 = stepsize:stepsize:1-stepsize;
y0 = stepsize:stepsize:1-stepsize;
[A, B] = meshgrid(x0, y0);
for i=1:size(A, 1)
    for j=1:size(A, 2)
        out(i,j) = eta(find(A(i,j)<=cumweights,1), find(B(i,j)<=cumweights,1));
    end
end
