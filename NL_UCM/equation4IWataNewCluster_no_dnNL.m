%% Second factor parameters depicted in (4)
function [apNew,bpNew,mu_jNew,invCjNew] = equation4IWataNewCluster_no_dnNL(params,W,X,d,n)
r = params.r;
K = params.K;
D = params.D;
invCjNew = zeros(K,K);
mu_jNew = zeros(K,1);

apNew = 0;
for id = 1:D
    if id == d
        apNew = apNew + params.Md(id)*(params.Nd(id)-1);
    else
        apNew = apNew + params.Md(id)*(params.Nd(id)-1);
    end
end
apNew = apNew*0.5 + params.a;
% For covariances Cj^-1
Wd = W{d};
% For the Kernelized version compute Kd
kern = kernCreate(Wd'*Wd,'rbf');
kern.variance = 0.01;
kern.inverseWidth = 1/(0.1)^2;
Kd = kernCompute(kern,Wd');
invCjNew = Kd+r*eye(K);
% For means <mu>
xdn = X{d}(n,:)';
kern = kernCreate(W{d}'*xdn,'rbf');
kern.variance = 0.01;
kern.inverseWidth = 1/(0.1)^2;
kd = kernCompute(kern,Wd',xdn');
mu_jNew = invCjNew\kd;


% For b' eq(6)
aux_xdn = 0;
for id = 1:D
    Xd = [X{id}]';
    [~,Nd] = size(Xd);
    
    for in = 1:Nd
        xdn = Xd(:,in);
        kern = kernCreate(xdn'*xdn,'rbf');
        kern.variance = 0.01;
        kern.inverseWidth = 1/(0.1)^2;
        if (in~=n)&&(id~=d)
            kxx = kernCompute(kern,xdn');
            aux_xdn = aux_xdn +kxx;
        end
    end
end
aux_muCj = 0;
for j = 1:params.J+1
    if j < params.J
        aux_muCj = aux_muCj + (params.mu_j(:,j)'*params.invCj(:,:,j)*params.mu_j(:,j));
    else
        aux_muCj = aux_muCj + (mu_jNew'*invCjNew*mu_jNew);
    end
    
end
bpNew = params.b + (1/2)*aux_xdn-(1/2)*aux_muCj;
