function ll = log_likelihoodEq4(S,params)
% for equation 3
J = params.J;
gammV = params.gamma;

auxNj = 0;
for j = 1:J
    Nj = Ndot_j(S,j,params);
    auxNj = auxNj + sum(log(1:(Nj-1)));
end

N = sum(params.Nd);
auxGammaN =  sum(log(params.gamma+0:(N-1)));

probS = (params.J)*log(gammV)+auxNj - auxGammaN;
% for equation 4
probNew = ((-params.auxSumD)*log(2*pi))+(log(params.r)*(0.5*params.K*params.J))+((params.a*log(params.b))-...
                (params.ap*log(params.bp)))+gammaln(params.ap)-gammaln(params.a);
            
auxDetCj = 0;
for i = 1:params.J
%     auxDetCj = auxDetCj+(log(det(inv(params.invCj(:,:,i))))*(0.5));
    auxDetCj = auxDetCj+(log(1/det(params.invCj(:,:,i)))*(0.5));
end

% ll = probNew+auxDetCj;
ll = probS+probNew+auxDetCj;