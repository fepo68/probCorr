function [probNew,ap_sdn_j,bp_sdn_j, mu_j_sdn_j, invCj_sdn_j] = probEq13NewCluster(params,W,X,d,n)

[apNew,bpNew,mu_jNew,invCjNew] = equation4IWataNewCluster_no_dn(params,W,X,d,n);

xdn = X{d}(n,:)';

ap_sdn_j = apNew; % eq 14
invCj_sdn_j = W{d}'*W{d} + invCjNew; % eq 17
mu_j_sdn_j = invCj_sdn_j\(W{d}'*xdn+invCjNew*mu_jNew);
bp_sdn_j = bpNew +0.5*xdn'*xdn+0.5*mu_jNew'*invCjNew*mu_jNew...
    -0.5*mu_j_sdn_j'*invCj_sdn_j*mu_j_sdn_j;


probNew = ((2*pi)^(-0.5*params.Md(d)))*(params.r^(0.5))*exp(((apNew*log(bpNew))-...
    (ap_sdn_j*log(bp_sdn_j))))*exp(gammaln(ap_sdn_j)-gammaln(apNew))*...
    ((det(inv(invCj_sdn_j))^0.5)/(det(inv(invCjNew))^0.5));
