function ll = likelihoodEq4(params,W,X)

probNew = ((2*pi)^(-0.5*params.auxSumD))*(params.r^(0.5*params.K*params.J))*exp(((params.a*log(params.b))-...
                (params.ap*log(params.bp))))*exp(gammaln(params.ap)-gammaln(params.a));
            
auxDetCj = 1;
for i = 1:params.J
    auxDetCj = auxDetCj*(det(inv(params.invCj(:,:,i)))^(0.5));
end

ll = probNew*auxDetCj;