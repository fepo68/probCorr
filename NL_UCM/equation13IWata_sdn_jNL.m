%% Second factor parameters depicted in (4) but excluding dn object
% see bottom factor of the equation 13
function [ap,bp,mu_j,invCj] = equation13IWata_sdn_jNL(X,S,W,params,de,ne,j)
a = params.a;
b= params.b;
r = params.r;
gammaVal = params.gamma;
K = params.K;
J = params.J;
D = params.D;
auxN = params.auxN;

linearVersion = false;

% Parameters stimation
ap = params.ap;
invCj = zeros(K,K,J);
mu_j = zeros(K,J);
bp = zeros(J,1);
if linearVersion == true
    
    for j = 1:J
        % For covariances C_{j,s_dn=j}^-1
        auxCj = zeros(K);
        [~,bp_no_dn,mu_j_no_dn,invCj_no_dn] = ...
            equation13IWata_excluding_dn(X,S,W,params,de,ne);
        invCj(:,:,j) = W{de}'*W{de}+invCj_no_dn(:,:,j);
        mu_j(:,j) = inv(invCj(:,:,j))*(W{de}'*[X{de}(ne,:)]'+ ...
            invCj_no_dn(:,:,j)*mu_j_no_dn(:,j));
        bp(j) = bp_no_dn + 0.5*[X{de}(ne,:)]*[X{de}(ne,:)]'+...
            0.5*mu_j_no_dn(:,j)'*invCj_no_dn(:,:,j)*mu_j_no_dn(:,j)-0.5*mu_j(:,j)'*invCj(:,:,j)*mu_j(:,j);
        
    end
else
    for j = 1:J
        % For covariances C_{j,s_dn=j}^-1
        auxCj = zeros(K);
        [~,bp_no_dn,mu_j_no_dn,invCj_no_dn] = ...
            equation13IWata_excluding_dnNL(X,S,W,params,de,ne);
        
        Wd = W{de};
        % For Cj,sdn_j
        kern = kernCreate(Wd'*Wd,'rbf');
        kern.variance = 0.01;
        kern.inverseWidth = 1/(0.1)^2;
        Kd = kernCompute(kern,Wd');
        invCj(:,:,j) = Kd+invCj_no_dn(:,:,j);
        % For mu_j,sdn_j
        xdn = [X{de}(ne,:)]';
        kern = kernCreate(Wd'*xdn,'rbf');
        kern.variance = 0.01;
        kern.inverseWidth = 1/(0.1)^2;
        kd = kernCompute(kern,Wd',xdn');
        mu_j(:,j) = invCj(:,:,j)\(kd+ ...
            invCj_no_dn(:,:,j)*mu_j_no_dn(:,j));
        % For bp_j,sdn_j
        kern = kernCreate(xdn'*xdn,'rbf');
        kern.variance = 0.01;
        kern.inverseWidth = 1/(0.1)^2;
        kxx = kernCompute(kern,xdn');
        bp(j) = bp_no_dn + 0.5*kxx+...
            0.5*mu_j_no_dn(:,j)'*invCj_no_dn(:,:,j)*mu_j_no_dn(:,j)-0.5*mu_j(:,j)'*invCj(:,:,j)*mu_j(:,j);
        
        
    end
end