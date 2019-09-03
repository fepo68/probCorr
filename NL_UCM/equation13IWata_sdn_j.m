%% Second factor parameters depicted in (4) but excluding dn object
% see bottom factor of the equation 13
function [ap,bp,mu_j,invCj] = equation13IWata_sdn_j(X,S,W,params,de,ne,j)
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
    % For covariances C_{j,s_dn=j}^-1
    auxCj = zeros(K);
    bp_no_dn = params.bp_no_dn;
    mu_j_no_dn = params.mu_j_no_dn;
    invCj_no_dn = params.invCj_no_dn;
%     [~,bp_no_dn,mu_j_no_dn,invCj_no_dn] = ...
%         equation13IWata_excluding_dn(X,S,W,params,de,ne);
%% Linear version
    invCj(:,:,j) = W{de}'*W{de}+invCj_no_dn(:,:,j);
    mu_j(:,j) = (invCj(:,:,j))\(W{de}'*[X{de}(ne,:)]'+ ...
        invCj_no_dn(:,:,j)*mu_j_no_dn(:,j));
    bp(j) = bp_no_dn + 0.5*[X{de}(ne,:)]*[X{de}(ne,:)]'+...
        0.5*mu_j_no_dn(:,j)'*invCj_no_dn(:,:,j)*mu_j_no_dn(:,j)-0.5*mu_j(:,j)'*invCj(:,:,j)*mu_j(:,j);
