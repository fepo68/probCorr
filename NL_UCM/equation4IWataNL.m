%% Second factor parameters depicted in (4)
function [ap,bp,mu_j,invCj] = equation4IWataNL(X,S,W,params)
a = params.a;
b= params.b;
r = params.r;
gamma = params.gamma;
K = params.K;
J = params.J;
D = params.D;
auxN = params.auxN;

linearVersion = false;

ap = a + auxN/2;
invCj = zeros(K,K,J);
mu_j = zeros(K,J);
for j = 1:J
    % For covariances Cj^-1
    auxCj = zeros(K);
    for d = 1:D
        
        % Linear version
        auxS = S{d};
        [~,n2]=find(auxS ==j);
        Ndj = sum(n2);
        Wd = W{d};
        
        if linearVersion == true
            auxCj = auxCj + Ndj*(Wd'*Wd);
        else
            % For the Kernelized version compute Kd
            kern = kernCreate(Wd'*Wd,'rbf');
            kern.variance = 0.01;
            kern.inverseWidth = 1/(0.1)^2;
            Kd = kernCompute(kern,Wd');
            auxCj = auxCj + Ndj*Kd;
        end
        
        
    end
    invCj(:,:,j) = auxCj+r*eye(K);
    % For means <mu>
    aux_mj = zeros(K,1);
    for d = 1:D
        auxS = S{d};
        [n1,n2]=find(auxS ==j); % Find objects that
        % has assigned the j-th cluster
        Wd = W{d};
        Xd = [X{d}]'; % Because objects are Md*Nd
        if linearVersion == true
            aux_xdn =sum(Xd(:,n1),2);
            aux_mj = aux_mj + Wd'*aux_xdn;
        else
            auxkd = zeros(K,1);
            if ~isempty(n1)
            xdn = Xd(:,n1(1));
            kern = kernCreate(Wd'*xdn,'rbf');
            kern.variance = 0.01;
            kern.inverseWidth = 1/(0.1)^2;
            
            for imu = 1: length(n2)
                % For the Kernelized version compute Kd
                xdn = Xd(:,n1(imu));
                kd = kernCompute(kern,Wd',xdn');
                auxkd = auxkd + kd;
            end
            end
            aux_mj = aux_mj + auxkd;
        end
    end
    mu_j(:,j) = invCj(:,:,j)\aux_mj;
    
end

% For b' eq(6)
aux_xdn = 0;
for d = 1:D
    Xd = [X{d}]';
    [Md,Nd] = size(Xd);
    
    for n = 1:Nd
        xdn = Xd(:,n);
        kern = kernCreate(xdn'*xdn,'rbf');
        kern.variance = 0.01;
        kern.inverseWidth = 1/(0.1)^2;
        if linearVersion == true
            aux_xdn = aux_xdn +(xdn'*xdn);
        else
            kxx = kernCompute(kern,xdn');
            aux_xdn = aux_xdn +kxx;
        end
    end
end
aux_muCj = 0;
for j = 1:J
    aux_muCj = aux_muCj + (mu_j(:,j)'*invCj(:,:,j)*mu_j(:,j));
end
bp = b + (1/2)*aux_xdn-(1/2)*aux_muCj;