%% Second factor parameters depicted in (4)
function [apNew,bpNew,mu_jNew,invCjNew] = equation4IWataNewCluster(params,W,X,d,n)
r = params.r;
K = params.K;
D = params.D;
invCjNew = zeros(K,K);
mu_jNew = zeros(K,1);

apNew = params.ap;
% For covariances Cj^-1
Wd = W{d};
invCjNew = (Wd'*Wd)+r*eye(K);
% For means <mu>
xdn = X{d}(n)';
mu_jNew = invCjNew\(Wd'*xdn);


% For b' eq(6)
aux_xdn = 0;
for id = 1:D
    Xd = [X{id}]';
    [~,Nd] = size(Xd);
    
    for in = 1:Nd
        xdn = Xd(:,in);
        if (in~=n)&&(id~=d)
            aux_xdn = aux_xdn +(xdn'*xdn);
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
