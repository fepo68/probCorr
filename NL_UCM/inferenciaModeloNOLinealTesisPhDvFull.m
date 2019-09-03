%% Unsupervised Clustering Matching Via LVM

% Input: multiple domain datasets X,initial number of clusters J, hyperparameters a, b, r, alpha , iteration number T
% Output: cluster assignments S, projection matrices W
clear all
close all
clc
load dataToyNLmodel26sept.mat
D = length(X);
T = 5; % iterations
% J = 2; % initial number of clusters J
alphaW = 0.05; % small variance for projection Matrices Wd
% K = 2; % dimensionality of the latent vector


% model parameters
a = 1;
b = 1;
r = 1;
gammaVal = 1;

N = 0;
auxN = 0;
for d = 1:D
    [Nd,Md] = size(X{d});
    N = N + Nd;
    auxN = auxN +(Nd*Md);
end
% Group all model parameters
params.Md = Md;
params.Nd = Nd;
params.a = a;
params.b = b;
params.r = r;
params.gamma = gammaVal;
params.K = K;
params.J = 1;
params.D = D;
params.N = N;
params.S = S;
params.auxN = auxN;
numOfClasses      = 1; % star with J classes (correspondence)

%% Inference

% Start the cluster asignement S and projection matrices W
S = {}; % Cluster assignements
W = {}; % Projection matrices
for d = 1:D
    [Nd,Md] = size(X{d});
    params.Md(d) = Md;
    s_dn = zeros(Nd,1);
    for n = 1:Nd
        s_dn (n) = randi(params.J);
    end
    S{d} = s_dn;
    [Nd,Md] = size(X{d});
    W{d} = alphaW*rand(Md,K);
    cs{d} = S{d};
    numOfPointsInClass = hist(cs{d},1:numOfClasses);
end

% Sample alpha from prior
alpha = gamrnd(params.a,params.b);
params.alpha = alpha;
% Sample Z from prior
z_j = mvnrnd(zeros(params.K,1),((alpha*r)^(-1))*eye(params.K));
% for i = 1:numOfClasses
%     for d = 1:D
%         Wd = W{d};
% %         mus(i,:,d)               = [Wd*z_j']'; % eq 1
%     end
% %     precs(:,:,i) = (alpha^(-1))*eye(params.K);
% end

for t = 1:T
    fprintf('Processing iteration: %d\n',t);
    %% Second factor parameters depicted in (4)
    [ap,bp,mu_j,invCj] = equation4IWata(X,S,W,params);
    params.ap = ap;
    params.bp = bp;
    params.mu_j = mu_j;
    params.invCj = invCj;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1: numOfClasses
        % sample class parameters from posteriors Eq 9 and 10 Iwata
        % draw the precision parameter alpha
        params.posalpha = gamrnd(ap,bp);
        % draw the posterior for the latent vector z_j
        params.z_j(:,i) = mvnrnd(mu_j(:,i),invCj(:,:,i));
    end
    %% E-step
    numOfDataPoints = zeros(1,D);
    numOfDimensions = zeros(1,D);
    for d = 1:D
        [Nd,~] = size(X{d});
        Wd = W{d};
        Xd = X{d};
        numOfDataPoints(d) = length(Xd);
        auxSumD = 0;
        for id = 1:D
            [Nd,Md] = size(X{d});
            auxSumD = Nd*Md+auxSumD;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % assign classes to data
        % precompute randomnumbers
        normRandVar = mvnrnd(Wd*params.z_j,params.alpha*eye(params.Md(d)),numOfDataPoints(d)); % eq 2
        randVar     = rand(1,numOfDataPoints(d));
        numOfDataPoints(d) = Nd;
        numOfDimensions(d) = Md;
        for n = 1:Nd
            %% Inference part I
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Map observation to the latent space z_j=inv(Wd'*Wd)*Wd'*x_dn
            zhat = inv(Wd'*Wd)*Wd'*[Xd(n,:)]';
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            data = Xd;
            % draw centroids and precisions from the prior
            permIx            = randperm(numOfDataPoints(d));
            for i = 1:numOfClasses
                % Draw z_j from priors
                z_j(i,:) =  mvnrnd(zeros(params.K,1),((params.alpha*params.r)^(-1))*eye(K));
                mus(i,:,d) = [Wd*z_j']';
                precs(:,:,i) = gamrnd(ap,bp)*eye(params.K); % (alpha^-1)*I
            end
            % for Wd*zj - eq 4
            % numOfClasses -> Clusters J
            prodC_j = 1;
            for i = 1:numOfClasses
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                % Posterior for the precision parameter alpha and the laten
                % vector z_j                eq - 9 and 10 Iwatta
                pos_alpha = gamrnd(params.ap,params.bp);
                pos_zj(i,:) = mvnrnd(params.mu_j(:,i),(r^-1)*inv(params.invCj(:,:,i)));
                % for mu - eq 4
                newLoc   = Wd*[pos_zj(i,:)]';
                newScale = pos_alpha;
                mus(i,:,d) = mvnrnd(newLoc',(newScale^-1)*eye(Md),1);
                
                
                % for s - eq 8
                precs(:,:,i) = (pos_alpha)*eye(Md);
                detPrecs(i) = det(precs(:,:,i));
                
                % objects assigned to cluster j, Iwata
                aux = 0;
                for id = 1:D
                    aux = aux + length(find(S{id}==i));
                end
                N_dotj(i) = aux;
                % prod over all C_j
                prodC_j = (det(inv(params.invCj(:,:,i)))^0.5)*prodC_j;
                
            end
            % for S - eq 3 iwata is analog to variable c_i in
            % rasmussen99 cluster asignement S
            pS_gamma = ((params.gamma^numOfClasses)*prod(factorial(N_dotj-1)))/...
                (params.gamma*prod(params.gamma+linspace(1,N-1,N-1)));
            % for probability of data X eq - 4 Iwata
            
            pX_SWabr = (2*pi)^(auxSumD/2)*params.r^((params.K*numOfClasses)/2)*...
                (params.b^params.a)/(params.bp^params.ap)*(gamma(params.ap)/gamma(params.a))*...
                prodC_j;
            %% Inference part II
            % sample sdn using probability p( j|X, S\dn,W, a, b, r, alpha ) (11)
            % from j = 1, . . . , J + 1
            %             s_dn (n) = randi(J); %prueba
            [params.ap_no_dn,params.bp_no_dn,params.mu_j_no_dn,params.invCj_no_dn] = ...
                equation13IWata_excluding_dn(X,S,W,params,d,n);
            
            prob = NaN*zeros(1,numOfClasses+1);
            probEq14 = NaN*zeros(1,numOfClasses+1);
            for j = 1:numOfClasses
                [ap_sdn_j,bp_sdn_j,mu_j_sdn_j,invCj_sdn_j] = ...
                    equation13IWata_sdn_j(X,S,W,params,d,n,i);
                % nij is N_{j\dn} in our model
                if cs{d}(i) == j;
                    nij               = numOfPointsInClass(j) - 1;
                else
                    nij               = numOfPointsInClass(j);
                end
                % Eq 14 - iwata
                Sno_dn = S{d};
                Sno_dn(n) = [];
                Jeq13 = length(unique(Sno_dn));
                probEq14(j) = ((2*pi)^(-0.5*params.Md(d)))*(r^(0.5*(j>Jeq13)))*((params.bp_no_dn^params.ap_no_dn)/...
                    (bp_sdn_j(j)^ap_sdn_j))*(gamma(ap_sdn_j)/gamma(params.ap_no_dn))*...
                    ((det(invCj_sdn_j(:,:,j))^0.5)/(det(params.invCj_no_dn(:,:,j))^0.5));
                %                 if nij > 0
                
                prob(j) = (nij)/(params.N-1+params.gamma)*probEq14(j); % Eq 12 iwata
                
                %                 else
                
                %                 end
                
            end
            % Now calculate the prior and likelihood for a new cluster
            xdn = X{d}(n,:);
            probNew = probEq4NewCluster(xdn,params,W,X);
            % Eq 14 - iwata
            
            prob(end) = (params.gamma)/(params.N-1+params.gamma)*probNew; % Eq 2 iwata
            
            cdf    = cumsum(prob);
            rndNum = randVar(i)*cdf(end);
            for j = 1:numOfClasses + 1
                if cdf(j) >= rndNum
                    
                    numOfPointsInClass(cs{d}(i)) = numOfPointsInClass(cs{d}(i)) - 1;
                    
                    if numOfPointsInClass(cs{d}(i)) < 1
                        % fprintf(['Removing class # ' num2str(cs(i)) '\n']);
                        % remove parameters
                        mus(cs{d}(i),:)              = [];
                        precs(:,:,cs{d}(i))          = [];
                        detPrecs(cs{d}(i))       = [];
                        numOfPointsInClass(cs{d}(i)) = [];
                        numOfClasses              = numOfClasses - 1;
                        j(j>cs{d}(i))                = j - 1;
                        % rename all higher classes
                        cs{d}(cs{d}>cs{d}(i))              = cs{d}(cs{d}>cs{d}(i)) - 1;
                    end
                    cs{d}(i) = j;
                    break;
                    
                end
            end
            
            if j == numOfClasses + 1
                % fprintf(['Adding new class # ' num2str(j) '\n']);
                % add parameters
                %                 mus(j,:)              = mu_new;
                %                 precs(:,:,j)          = s_new;
                %                 detPrecs(:,:,j)       = det(s_new);
                
                % Posterior for the precision parameter alpha and the laten
                % vector z_j                eq - 9 and 10 Iwatta
                params.J = params.J+1;
                params.S{d}(n) = params.J;
                S = params.S;
                [ap,bp,mu_j,invCj] = equation4IWata(X,S,W,params);
                params.ap = ap;
                params.bp = bp;
                params.mu_j = mu_j;
                params.invCj = invCj;
                
                pos_alpha = gamrnd(ap,bp);
                pos_zj(j,:) = mvnrnd(params.mu_j(:,j),(pos_alpha^-1)*inv(params.invCj(:,:,j)));
                % for mu - eq 4
                newLoc   = Wd*[pos_zj(j,:)]';
                newScale = pos_alpha;
                mus(j,:) = mvnrnd(newLoc',(newScale^-1)*eye(Md),1);
                
                
                % for s - eq 8
                precs(:,:,j) = (pos_alpha)*eye(Md) + 1e-10*eye(numOfDimensions);
                detPrecs(j) = det(precs(:,:,j));
                %                 params.mu_j(j,:) = [mus(j,:)]';
                %                 params.invCj(:,:,i) =
                
                
                
                numOfPointsInClass(j) = 1;
                numOfClasses          = numOfClasses + 1;
            else
                numOfPointsInClass(j) = numOfPointsInClass(j) + 1;
            end
            
            %             if sdn == (J+1)
            %                 %                 update the number of clusters J = J + 1
            %             end
        end
    end
    %% M-step
    for d = 1:D
        %         update Wd using a numerical optimization method using (18)
        W{d} = MstepJointLLexact(X,S,params,d)
    end
end



