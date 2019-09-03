%% Likelihood optimization equation 4 Iwatta

function [f,g] = prjectionWOptim(x,params,W,X,S,d)
% Data setup
% [m,n] = size(Data.Wd);
% S = params.S;
% X = params.X;
% d = params.d;
Saux = S{d};
Xd = X{d};
Wd = reshape(x,params.Md(d),params.K);
W{d} = Wd;
k =rank(Wd);


% Function Value 
gammaVal = params.gamma;
J = params.J;
K = params.K;

auxNj = 0;


%% Second factor parameters depicted in (4)
[ap,bp,mu_j,invCj] = equation4IWata(X,S,W,params);
params.ap = ap;
params.bp = bp;
params.mu_j = mu_j;
params.invCj = invCj;
f = log_likelihoodEq4(params);
g = gradientLLEq18(X,W,S,params,d);
