%% Unsupervised Clustering Matching Via LVM

% Input: multiple domain datasets X,initial number of clusters J, hyperparameters a, b, r, alpha , iteration number T
% Output: cluster assignments S, projection matrices W
clear all
close all
clc
load toy10Nd3Md2K5J.mat

D = length(X);
[Nd,Md] = size(X{1});

T = 500; % iterations
J = 5; % initial number of clusters J
alphaW = 0.05; % small variance for projection Matrices Wd
K = 2; % dimensionality of the latent vector
S = {}; % Cluster assignements
W = {}; % Projection matrices
for d = 1:D
    s_dn = zeros(Nd,1);
    for n = 1:Nd
        s_dn (n) = randi(J);
    end
    S{d} = s_dn;
    [~,Md] = size(X{d});
    W{d} = alphaW*rand(Md,K);
end

% model parameters
a = 1;
b = 1;
r = 1;
gamma = 1;

N = 0;
auxN = 0;
for d = 1:D
    [Nd,Md] = size(X{d});
    N = N + Nd;
    auxN = auxN +(Nd*Md);
end
% Group all model parameters
params.a = a;
params.b = b;
params.r = r;
params.gamma = gamma;
params.K = K;
params.J = J;
params.D = D;
params.auxN = auxN;

for t = 1:T
    %% Second factor parameters depicted in (4)
    [ap,bp,mu_j,invCj] = equation4IWata(X,S,W,params);
    %% E-step
    for d = 1:D
        [Nd,~] = size(X{d});
        for n = 1:Nd
            % sample sdn using probability p( j|X, S\dn,W, a, b, r, alpha ) (11)
            % from j = 1, . . . , J + 1
            s_dn (n) = randi(J); %prueba
            if sdn == (J+1)
                %                 update the number of clusters J = J + 1
            end
        end
    end
    %% M-step
    for d = 1:D
        %         update Wd using a numerical optimization method using (18)
    end
end



