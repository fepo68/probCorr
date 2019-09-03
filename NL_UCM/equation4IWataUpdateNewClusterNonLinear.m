%% Second factor parameters depicted in (4)
function [bp] = equation4IWataUpdateNewClusterNonLinear(X,S,W,params)
a = params.a;
b= params.b;
r = params.r;
gamma = params.gamma;
K = params.K;
J = params.J;
D = params.D;
auxN = params.auxN;

invCj = params.invCj;
mu_j = params.mu_j;

% ap = a + auxN/2;
% invCj = zeros(K,K,J);
% mu_j = zeros(K,J);
% for j = 1:J
%     For covariances Cj^-1
%     auxCj = zeros(K);
%     for d = 1:D
%         auxS = S{d};
%         [~,n2]=find(auxS ==j);
%         Ndj = sum(n2);
%         Wd = W{d};
%         auxCj = auxCj + Ndj*(Wd'*Wd);
%     end
%     invCj(:,:,j) = auxCj+r*eye(K);
%     For means <mu>
%     aux_mj = zeros(K,1);
%     for d = 1:D
%         auxS = S{d};
%         [n1,n2]=find(auxS ==j); % Find objects that
%         has assigned the j-th cluster
%         Wd = W{d};
%         Xd = [X{d}]'; % Because objects are Md*Nd
%         aux_xdn =sum(Xd(:,n1),2);
%         aux_mj = aux_mj + Wd'*aux_xdn;
%     end
%     mu_j(:,j) = invCj(:,:,j)\aux_mj;
%
% end

% For b' eq(6)

aux_xdn = 0;

for d = 1:D
    Xd = [X{d}]';
    [Md,Nd] = size(Xd);
        
    %% Linear form 
%     aux = sum(diag((Xd'*Xd));
%     aux_xdn =aux_xdn + aux;
    %% Compute the kernelized form of mu_j
    %     kern = kernCreate(xdn,params.kernType);
    %     kern.variance = params.varianceKxx;
    %     kern.inversewidth = params.inversewithKxx;
    %     tic
    %     aux_xdn = aux_xdn +sum(diag(Xd'*Xd));
    %     toc
    %         tic    
    kern.type = params.kernType;
    kern.variance = params.varianceKxx;
    kern.inverseWidth = params.inversewithKxx;
    Kdxx = mykernCompute(kern,Xd);
    aux_xdn = aux_xdn +sum(diag(Kdxx));
%     for n = 1:Nd
%         xdn = Xd(:,n);
%         
%         %         kern.inversewidth = 10;
%         %         kxx = kernCompute(kern,xdn');
%         kxx = mykernCompute(kern,xdn);
%         %         aux_xdn = aux_xdn +(xdn'*xdn);
%         aux_xdn = aux_xdn +kxx;
%     end
%     %         toc
end
aux_muCj = 0;
for j = 1:J
    aux_muCj = aux_muCj + (mu_j(:,j)'*invCj(:,:,j)*mu_j(:,j));
end
bp = b + (1/2)*aux_xdn-(1/2)*aux_muCj;