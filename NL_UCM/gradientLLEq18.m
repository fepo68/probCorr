function g = gradientLLEq18(x,X,W,S,params,d)

Wd = reshape(x',params.Md(d),params.K);
W{d} = Wd;

aux1 = zeros(params.K);
aux2 = zeros(params.Md(d),params.K);
auxS = S{d};
Wd = W{d};
Xd = X{d};
I = eye(params.K);
for j = 1:(params.J)
    [valP,valI] = find(auxS == j);
    Ndj = length(valP);
    aux1 = Ndj*((params.invCj(:,:,j))\I) + aux1;
%     aux1 = Ndj*(inv(params.invCj(:,:,j))) + aux1;
    
    sumAux = sum(Xd(valP,:)',2)*params.mu_j(:,j)';
    aux2 = (Ndj*Wd*(params.mu_j(:,j)*params.mu_j(:,j)') - sumAux) + aux2;
end


g = Wd*aux1 + (params.ap / params.bp) * aux2;
g = g(:)';