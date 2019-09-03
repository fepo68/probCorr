function probNew = probEq13NewClusterNL(params,W,X,d,n)

[apNew,bpNew,mu_jNew,invCjNew] = equation4IWataNewCluster_no_dnNL(params,W,X,d,n);

xdn = X{d}(n,:)';

ap_sdn_j = apNew; % eq 14
% For Cj,sdn_j
kern = kernCreate(W{d}'*W{d},'rbf');
kern.variance = 0.01;
kern.inverseWidth = 1/(0.1)^2;
Kd = kernCompute(kern,W{d}');
invCj_sdn_j = Kd + invCjNew; % eq 17
% For mu_j,sdn_j
kern = kernCreate(W{d}'*xdn,'rbf');
kern.variance = 0.01;
kern.inverseWidth = 1/(0.1)^2;
kd = kernCompute(kern,W{d}',xdn');
mu_j_sdn_j = invCj_sdn_j\(kd+invCjNew*mu_jNew);
% For bp
kern = kernCreate(xdn'*xdn,'rbf');
kern.variance = 0.01;
kern.inverseWidth = 1/(0.1)^2;
kxx = kernCompute(kern,xdn');
bp_sdn_j = bpNew +0.5*kxx+0.5*mu_jNew'*invCjNew*mu_jNew...
    -0.5*mu_j_sdn_j'*invCj_sdn_j*mu_j_sdn_j;


probNew = ((2*pi)^(-0.5*params.Md(d)))*(params.r^(0.5))*((bpNew^apNew)/...
    (bp_sdn_j^ap_sdn_j))*(gamma(ap_sdn_j)/gamma(apNew))*...
    ((det(inv(invCj_sdn_j))^0.5)/(det(inv(invCjNew))^0.5));
