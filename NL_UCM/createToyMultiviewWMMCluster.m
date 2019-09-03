function [Sclust,Y,Fd,X] = createToyMultiviewWMMCluster(Nd,Dd,V,K,Q,beta)

% Draw samples from the Multiview Warped Mixture Model
% Inputs:
% Nd -> array where each element contains the number of objects per view
% Dd -> array where each element contains the number of outputs per view
% V -> Number of views
% K -> Number of Clusters
% Q -> Dimensionality of the Latent Space
doJitter = true;
GEM = false;
% 1. Draw mixture weights
if GEM == true
    % Initialize lambda with a Stic Breaking Process
else
    % All mixture weigths have the same probability
    lambda = ones(1,K)/K;
end
% 2. For each component c = 1,..., infty
R = zeros(Q,Q,K);
mu = zeros(Q,K);
S = eye(Q);
nu = 10; % degree of freedom for the Wishart dist
u = rand(Q,K);
r = 0.05;
for k = 1:K
    % Draw preciosion Rc
    R(:,:,k) = wishrnd(S,nu);
    % Draw mean
    mu_aux = mvnrnd(u(:,k),(r*R(:,:,k))\eye(Q));
    mu(:,k) = mu_aux';
    
end
Z = [];
X = {};
Sclust = {};
for v = 1:V
    Xd_aux = zeros(Nd(v),Q);
    Zv = mnrnd(1,lambda,Nd(v));
    [~, idZn] = max(Zv');
    Sclust{v} = idZn;
    for n = 1:Nd(v)
        
        xd = mvnrnd(mu(:,idZn(n)),inv(R(:,:,idZn(n))));
        Xd_aux(n,:) = xd;
    end
    X{v} = Xd_aux;
    %     [~,idZn] = max(Zv');
    %     idZn = idZn';
    %     Xd_aux = mvnrnd(mu(:,idZn),inv(R(:,:,idZn)),Nd(v));
    
    %% Draw functions
    if v == 1
        kern = kernCreate(Xd_aux, 'rbf');
        kern.inverseWidth = 0.1;
        kern.variance = 1;
    else
        kern = kernCreate(Xd_aux, 'rbfard');
        kern.inverseWidth = 0.01;
        kern.variance = 0.8;
    end
    Kmat = kernCompute(kern, Xd_aux);
    if doJitter == true
        [UC, jitter] = jitChol(Kmat);
        Kmat = UC'*UC;
    end
    figure,imagesc(Kmat);
    % need to take the real part of the sample as the kernel is numerically less than full rank
    % Sample fdm using a fdm(z)~GP(0,K)
    Fd{v} = real(gsamp(zeros(1, size(Xd_aux, 1)), Kmat, Dd(v)))';
%     for d = 1:Dd
%     end
end

% Draw Observations
Y = {};
for v = 1:V
    Fdaux = Fd{v};
    Yaux = zeros(Nd(v),Dd(v));
    for n = 1:Nd(v);
        Yaux(n,:) = mvnrnd(Fdaux(n,:),(beta^(-1))*eye(Dd(v)));
    end
    Y{v} = Yaux;
end

