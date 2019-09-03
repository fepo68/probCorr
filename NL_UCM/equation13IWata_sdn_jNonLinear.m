%% Second factor parameters depicted in (4) but excluding dn object
% see bottom factor of the equation 13
function [ap,bp,mu_j,invCj] = equation13IWata_sdn_jNonLinear(X,S,W,params,de,ne,j)
a = params.a;
b= params.b;
r = params.r;
gammaVal = params.gamma;
K = params.K;
J = params.J;
D = params.D;
auxN = params.auxN;



% Parameters stimation
ap = params.ap;
invCj = zeros(K,K,J);
mu_j = zeros(K,J);
bp = zeros(J,1);
% for j = 1:J
%% For covariances C_{j,s_dn=j}^-1
auxCj = zeros(K);
bp_no_dn = params.bp_no_dn;
mu_j_no_dn = params.mu_j_no_dn;
invCj_no_dn = params.invCj_no_dn;
%     [~,bp_no_dn,mu_j_no_dn,invCj_no_dn] = ...
%         equation13IWata_excluding_dn(X,S,W,params,de,ne);
%% Linear version
%     invCj(:,:,j) = W{de}'*W{de}+invCj_no_dn(:,:,j);
%     mu_j(:,j) = (invCj(:,:,j))\(W{de}'*[X{de}(ne,:)]'+ ...
%         invCj_no_dn(:,:,j)*mu_j_no_dn(:,j));
%     bp(j) = bp_no_dn + 0.5*[X{de}(ne,:)]*[X{de}(ne,:)]'+...
%         0.5*mu_j_no_dn(:,j)'*invCj_no_dn(:,:,j)*mu_j_no_dn(:,j)-0.5*mu_j(:,j)'*invCj(:,:,j)*mu_j(:,j);
%% Nonlinear version
% for C_j
kern.type = params.kernType;
kern.variance = params.varianceKww;
kern.inverseWidth = params.inversewithKww;
Kd = mykernCompute(kern,W{de});
invCj(:,:,j) = Kd+invCj_no_dn(:,:,j);
% for mu
kern.type = params.kernType;
kern.variance = params.varianceKwx;
kern.inverseWidth = params.inversewithKwx;
Kwx = mykernCompute(kern,W{de},[X{de}(ne,:)]');

mu_j(:,j) = (invCj(:,:,j))\(Kwx+ ...
    invCj_no_dn(:,:,j)*mu_j_no_dn(:,j));
% for bp
kern.type = params.kernType;
kern.variance = params.varianceKxx;
kern.inverseWidth = params.inversewithKxx;
kxx = mykernCompute(kern,[X{de}(ne,:)]');
bp(j) = bp_no_dn + 0.5*kxx+...
    0.5*mu_j_no_dn(:,j)'*invCj_no_dn(:,:,j)*mu_j_no_dn(:,j)-0.5*mu_j(:,j)'*invCj(:,:,j)*mu_j(:,j);

% end