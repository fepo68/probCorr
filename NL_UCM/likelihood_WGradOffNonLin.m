%% Likelihood optimization equation 4 Iwatta
function [f,g] = likelihood_WGradOffNonLin(x,params,W,X,S,d)
% Data setup
% [m,n] = size(Data.Wd);
% S = params.S;
% X = params.X;
% d = params.d;
if ~strcmp(params.kernType,'linear')
    params.varianceKxx = x(1);
    params.inversewithKxx = x(2);
    
    params.varianceKwx = x(3);
    params.inversewithKwx = x(4);
    
    params.varianceKww = x(5);
    params.inversewithKww = x(6);
    
    
    Wd = reshape(x(7:end)',params.Md(d),params.K);
else
    Wd = reshape(x',params.Md(d),params.K);
end
Saux = S{d};
Xd = X{d};
W{d} = Wd;
k =rank(Wd);


% Function Value
gammaVal = params.gamma;
J = params.J;
K = params.K;

auxNj = 0;


%% Second factor parameters depicted in (4)
[ap,bp,mu_j,invCj] = equation4IWataNonLinear(X,S,W,params);
params.ap = ap;
params.bp = bp;
params.mu_j = mu_j;
params.invCj = invCj;
f = -log_likelihoodEq4(S,params); % objetive function (ll)
% if nargout > 1 % gradient required
%     g = gradientLLEq18(x,X,W,S,params,d);
% end
% g = gradientLLEq18(x,X,W,S,params,d);
g = -mygradientCompute(x,X,W,S,params,d);
