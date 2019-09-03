%% Kernel Trick Embedded for GMM with multiple vies

function [pli,Ghat] = myKernelGMMCompute(X,K,T)
% s = RandStream('mt19937ar','Seed',1e5*3);
% RandStream.setGlobalStream(s);

% X : Observed Data NxD
% K : number of clusters
% T : number of it (max)
[N,D] = size(X);
kmeansini = true;
GMM = true;
eigsV = true;
versionGapp = true;
% Initialize pli
if kmeansini == true
    %K-by-D matrix indicating the choosing of the initial K centroids.
    %we use kmeans to extract the centroids
    ncentroids = K;
    centroids = zeros(ncentroids, D);
    %         Set up vector of options for kmeans trainer
    options = foptions;
    options(1)  = 1;		% Prints out error values.
    options(5) = 1;
    options(14) = 10;		% Number of iterations.
    [centroids,~ , pli] = kmeans(centroids, X,options);
    %         pli(pli == 1) = .7;
    %         pli(pli == 0) = (1-.7)/(K-1);
    
    
else
    if GMM ==true
        [pli, model] = gmmSingleView(X, K);
    else
        aux = (1/K)*ones(1,K);
        % pli = mnrnd(N,aux);
        pli = repmat(aux,N,1);
    end
end
alpha = zeros(1,K);
wli = zeros(N,K);
% dPhi = round(0.05*N);
dPhi = 4;
cte1 = (2*pi)^(dPhi/2);
% Since Kern is computed from the data
kern = kernCreate(X, 'poly');
% kern.inverseWidth = 1;
kern.inverseWidth = 1/0.1;
kern.variance = 1;
kern.degree = 2;
kern.weightVariance = 0.7;
Ker = kernCompute(kern, X);
E_N = (1/N)*ones(N,N);
Kc = Ker - E_N*Ker - Ker*E_N + E_N*Ker*E_N;
% Inference
for it = 1:T
    fprintf('Processing iteration #%d\n',it);
    %% 1. Compute alpha , wli , W, W' K(t) and K'(t)
    alpha = (1/N)*sum(pli);
    % Wli in a proper way (summing through N data points)
    wli = sqrt(pli./(repmat(sum(pli,1),N,1)));
    % wli as it  appears in the paper (sum over M gaussians)
    %     wli = sqrt(pli./(repmat(sum(pli,2),1,K)));
    Ghat = zeros(N,K);
    W = {};
    Wp = {};
    Kl = {};
    Klp = {};
    Kl_til = {};
    Klp_til = {};
    eigK = [];
    
    for m = 1:K
        W{m} = wli(:,m)*wli(:,m)';
        Wp{m} = ones(N,1)*wli(:,m)';
        
        
        % For Kl, and Kl_til
        auxW = wli*wli';
        Kl{m} = W{m}.*Ker;
%                 Kl_til{m} = W{m}*Kc;
        Kl_til{m} = Kl{m} - W{m}*Kl{m}-Kl{m}*W{m}...
            + W{m}*Kl{m}*W{m};
        %         Kl_til{m} = Kl{m} - (1/N)*W{m}*Kl{m}-(1/N)*Kl{m}*W{m}...
        %             +(1/N^2)*W{m}*Kl{m}*W{m};
        % For Klp, and Klp_hat
        
        auxW = repmat(wli(:,m),1,N);
        Klp{m} = Wp{m}.*Ker;
%                 Klp_til{m} = Wp{m}*Kc;
        Klp_til{m} = Klp{m} - Wp{m}*Kl{m}-Klp{m}*W{m}...
            + Wp{m}*Kl{m}*W{m};
        %         Klp_til{m} = Klp{m} - (1/N)*Wp{m}*Kl{m}-(1/N)*Klp{m}*W{m}...
        %             +(1/N^2)*Wp{m}*Kl{m}*W{m};
        %% Step 3. Compute the d_phi largest eig of Kl_tilde
        %         opts.tol = 1e-3;
        if eigsV == true
            [eVecs,eVals] = eigs(Kl_til{m},dPhi);
            eigK(m).D = real(eVals);
            eigK(m).V = real(eVecs);
            eVals = real(eVals);
            eVecs = real(eVecs);
        else
            [eVecsa,eValsa] = eig(Kl_til{m});
            eigK(m).D = real(eValsa(1:dPhi,1:dPhi));
            eigK(m).V = real(eVecsa(:,1:dPhi));
            eVecs = real(eVecsa(:,1:dPhi));
            eVals = real(eValsa(1:dPhi,1:dPhi));
        end
        %         [eVecs,eVals] = eigs(Kl_til{l},dPhi,'lm',opts);
        
        %% Step 4. Gaussian probability density Gl(phi(xj)|theta_l)
        if versionGapp == true
            % Compute rho
            rho_ast = (1/(N-dPhi))*((norm(Kl_til{m},'fro')^2)-sum(diag(eVals)));
            %         rho_ast = (1/(N-dPhi))*sum(diag(eVals));
            % Compute ye
            yeA = ((eVecs)'*(Klp_til{m}))';
            
%             yeA = zeros(N,dPhi);
%             for n = 1:N
%                 for e = 1:dPhi
%                     yeA(n,e) = eVecs(:,e)'*Klp_til{m}(:,n); % springer Version Jindong
%                 end
%             end
            %         ye = diag(eigK(l).V'*(Klp_til{l}));
            % Compute e^2
            %% Compute Kc for e
            Kc = (eye(N)-(1/N)*ones(N,1)*ones(N,1)')*Kl{m}*(eye(N)-(1/N)*ones(N,1)*ones(N,1)');
            e = diag(Kl_til{m}) - sum(yeA.^2,2); % kl_til
            %         e = diag(Kl_til{l}./W{l}) - sum(yeA.^2,2); % kl_til
            %         e = diag(Ker) - sum(yeA.^2,2); % k(xi,xj)
            prinSubS = (1/(((2*pi)^(dPhi/2))*prod(sqrt(diag(eVals)))))*...
                exp(-0.5*sum((yeA.^2)./repmat( diag(eVals)',N,1),2));
            ortSubS = (1/(2*pi*rho_ast)^((N-dPhi)/2))*exp(-(e)/(2*rho_ast));
            
%             Gl = prinSubS.*ortSubS;
            Gl = prinSubS; % only the principal subspace
            Gl(isinf(Gl)) = realmax;
        else
             yeA = zeros(N,dPhi);
            for n = 1:N
                for e = 1:dPhi
                    yeA(n,e) = eVecs(:,e)'*Klp{m}(:,n); % IEEE Version Jindong
                end
            end
            prodeVals = (2*pi*diag(eVals)).^(0.5)';
            argExp = exp(-0.5*((yeA.^2)./repmat( diag(eVals)',N,1)));
            Gl = prod(repmat(prodeVals,N,1).*argExp,2);
        end
        Ghat(:,m) = Gl;
        
        % compute in log version
        %         aux1 = -(dPhi/2)*log(2*pi)-sum(0.5*log(diag(eigK(l).D)))-0.5*sum((auxYe.^2)./repmat( diag(eigK(l).D),1,N));
    end
    % Compute posterior probabilities
    aux = repmat(sum(repmat(alpha,N,1).*Ghat,2),1,K);
    pli = (repmat(alpha,N,1).*Ghat)./aux;
    
    if K ~=2
        plis = sum(pli);
        val = find(plis<0.02*sum(plis));
        if length(val) == 1
            pli(:,val) = [];
            K = K - 1;
        else
            goodC = setdiff(1:K,val);
            cpli = pli;
            pli = zeros(N,length(goodC));
            for nrem = 1:length(goodC)
                pli(:,nrem) = cpli(:,goodC(nrem));
                
            end
            K = length(goodC);
            %         pause
        end
    end
    %% check for ensuring at least two clusters
    if K == 1
        % At least perform two partitions
        %K-by-D matrix indicating the choosing of the initial K centroids.
        %we use kmeans to extract the centroids
        K = 2;
        ncentroids = K;
        centroids = zeros(ncentroids, D);
        %         Set up vector of options for kmeans trainer
        options = foptions;
        options(1)  = 1;		% Prints out error values.
        options(5) = 1;
        options(14) = 10;		% Number of iterations.
        [centroids,~ , pli] = kmeans(centroids, X,options);
        
    end
    
    %% Search for empty clusters
    
    
    %     pli = real(pli);
    % %%     numerical problem
    %     % 1. Checking for NaN values
    %     posNaN = isnan(pli);
    %     aux = find(sum(posNaN,2) == 1);
    %     if ~isempty(aux)
    %
    %         for ll = 1:length(aux)
    %             pli(aux(ll),:) = zeros(1,K);
    %             pli(aux(ll),randi(K)) = 1;
    %         end
    %     end
    %     fix = sum(pli,2);
    %     [aa,~] = find(fix<1);
    %     [bb,~] = find(fix>1);
    %     if ~isempty(aa)
    %         %         [~,I] = min(pli(aa,:),[],2);
    %         [~,Pos] = max(pli(aa,:),[],2);
    %         pli(aa,:) = 0;
    %         for ii = 1:length(aa)
    %             pli(aa(ii),Pos(ii)) = 1;
    %         end
    %     end
    %     if ~isempty(bb)
    %         %         [~,I] = min(pli(bb,:),[],2);
    %         [valP,Pos] = max(pli(bb,:),[],2);
    %         pli(bb,:) = 0;
    %         delta = (1-valP);
    %         for ii = 1:length(bb)
    %             pli(bb(ii),Pos(ii)) = 1;
    %         end
    %     end
    %
    
    %         isnan(pli) == 1
    %% To constraint the wli values
    %     pli = fixPli(pli);
    
    
end