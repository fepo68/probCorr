function probNew = probEq4NewCluster(params,W,X,d,n)

[apNew,bpNew,mu_jNew,invCjNew] = equation4IWataNewCluster_no_dn(params,W,X,d,n);

ap_sdn_j = apNew;

Jnew = params.J + 1;

invCj_aux = params.invCj;
invCj_aux(:,:,end+1) = invCjNew;

probNew = ((2*pi)^(-0.5*params.auxN))*(params.r^(0.5*params.K*(Jnew)))*((params.b^params.a)/...
                (bpNew^params.ap))*(gamma(params.ap)/gamma(params.a));

detCj = 1;            
for i = 1:Jnew
    detCj = detCj*(det(invCj_aux(:,:,i))^0.5);    
end

probNew = probNew*detCj;