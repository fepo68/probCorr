function g = mygradientCompute(x,X,W,S,params,d)

dt = 1e-4;
% numerical gradient can be computed as (central gradient):
%  df    f(t + dt) - f(t-dt)
% ---- = -------------------
%  dt           2dt

g = zeros(length(x),1);

if strcmp(params.kernType,'linear')
    
    for i = 1:length(x)
        xn = x;
        xn(i) = x(i) + dt;
        Wd = reshape(xn',params.Md(d),params.K);
        W{d} = Wd;
        [ap,bp,mu_j,invCj] = equation4IWataNonLinear(X,S,W,params);
        params.ap = ap;
        params.bp = bp;
        params.mu_j = mu_j;
        params.invCj = invCj;
        fpdt = log_likelihoodEq4(S,params);
        
        xn = x;
        xn(i) = x(i)- dt;
        Wd = reshape(xn',params.Md(d),params.K);
        W{d} = Wd;
        [ap,bp,mu_j,invCj] = equation4IWataNonLinear(X,S,W,params);
        params.ap = ap;
        params.bp = bp;
        params.mu_j = mu_j;
        params.invCj = invCj;
        fmdt = log_likelihoodEq4(S,params);
        
        g(i) = (fpdt - fmdt)/(2*dt);
        
    end
end