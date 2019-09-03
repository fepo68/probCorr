%% Unsupervised Clustering Matching Via LVM

% Input: multiple domain datasets X,initial number of clusters J, hyperparameters a, b, r, alpha , iteration number T
% Output: cluster assignments S, projection matrices W
clear all
close all
clc

% load dataToyIwatamodel20nov2D4Ndj5J3K
% load dataToyIwatamodel20nov2D20Ndj5J3K % for synth3
% load dataToyIwatamodel30nov2D20Ndj5J5K % for synth5
% load dataToyIwatamodel30nov2D20Ndj5J3K
% load dataToyIwatamodel30nov2D400Ndj2J3K
% load dataToyIwatamodel30nov2D500Ndj2J3K
% load dataToyIwatamodel10Dic2D50Ndj2J3K
load dataToyIwatamodel18Ene162D50Ndj2J3K
% load dataIGMM2D

% X{1} = data;

D = length(X);
T = 100; % iterations
J = 1; % initial number of clusters J
alphaW = 0.1; % small variance for projection Matrices Wd
K = 3; % dimensionality of the latent vector

trueS = S;
trueW = W;
trueJ = J;
clear J;
% model parameters
a = 1;
b = 1;
r = 1;
gammaVal = 1;
K = 3;

optiW = false;

N = 0;
auxN = 0;
for d = 1:D
    [Nd,Md] = size(X{d});
    params.Md(d) = Md;
    params.Nd(d) = Nd;
    N = N + Nd;
    auxN = auxN +(Nd*Md);
end

auxSumD = auxN/2;

% Group all model parameters
% params.Md = Md;
% params.Nd = Nd;
params.a = a;
params.b = b;
params.r = r;
params.gamma = gammaVal;
params.K = K;
params.J = 1;
numOfClasses = params.J;
params.D = D;
params.N = N;

params.auxN = auxN;
params.auxSumD = auxSumD;
% numOfClasses      = 1; % star with J classes (correspondence)

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
    W{d} = alphaW*normrnd(0,1,[Md,params.K]);
    cs{d} = S{d};
    numOfPointsInClass = hist(cs{d},1:numOfClasses);
end
params.S = S;
% Sample alpha from prior
alpha = gamrnd(params.a,1/params.b);
params.alpha = alpha;
% Sample Z from prior
z_j = mvnrnd(zeros(params.K,1),((alpha*r)^(-1))*eye(params.K));
% for i = 1:numOfClasses
%     for d = 1:D
%         Wd = W{d};
%         mus(i,:,d)               = [Wd*z_j']'; % eq 1
%     end
%     precs(:,:,i) = (alpha^(-1))*eye(params.K);
% end

%% Second factor parameters depicted in (4)
[ap,bp,mu_j,invCj] = equation4IWata(X,S,W,params);
params.ap = ap;
params.bp = bp;
params.mu_j = mu_j;
params.invCj = invCj;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ll = zeros(1,T);
for t = 1:T
    fprintf('Processing iteration: %d\n',t);
    
    for i = 1: numOfClasses
        % sample class parameters from posteriors Eq 9 and 10 Iwata
        % draw the precision parameter alpha
        params.posalpha = gamrnd(params.ap,1/params.bp);
        % draw the posterior for the latent vector z_j
        params.z_j(:,i) = mvnrnd(params.mu_j(:,i),params.invCj(:,:,i));
    end
    %% E-step
    numOfDataPoints = zeros(1,D);
    numOfDimensions = zeros(1,D);
    for d = 1:D
        [Nd,Md] = size(X{d});
        Wd = W{d};
        Xd = X{d};
        Sd = S{d};
        numOfDataPoints(d) = length(Xd);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % assign classes to data
        % precompute randomnumbers
        %         normRandVar = mvnrnd(Wd*params.z_j,params.alpha*eye(params.Md(d)),numOfDataPoints(d)); % eq 2
        %         randVar     = rand(1,numOfDataPoints(d));
        numOfDataPoints(d) = Nd;
        numOfDimensions(d) = Md;
        for n = 1:Nd
            %% Inference part I
            % Compute de model parameters by excluding the x_dn object
            [params.ap_no_dn,params.bp_no_dn,params.mu_j_no_dn,params.invCj_no_dn] = ...
                equation13IWata_excluding_dn(X,S,W,params,d,n);
            
            % Remove the old cluster ssignement for x_dn and update
            % the parameters accordingly of the cluster it got removed from
            %             auxSval = S{d}(n);
            S{d}(n) = NaN;
            %             % we need to check the cluster assignement for the rest of the domains
            %             auxSd =[];
            %             for daux = 1:D
            %                 auxSd = [auxSd;S{d}];
            %             end
            %
            %             [auxP,auxI] = find(auxSd == auxSval);
            %             if isempty(auxI)
            %                 params.mu_j(:,auxSval) = [];
            %                 params.invCj(:,:,auxSval) = [];
            %                 params.J = params.J - 1;
            %                 numOfClasses = numOfClasses -1;
            %             end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Map observation to the latent space z_j=inv(Wd'*Wd)*Wd'*x_dn
            %             zhat = inv(Wd'*Wd)*Wd'*[Xd(n,:)]';
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % numOfClasses -> Clusters J
            prodC_j = 1;
            N_dotj =[];
            for i = 1:numOfClasses
                % objects assigned to cluster j, Iwata
                aux = 0;
                for id = 1:D
                    aux = aux + length(find(S{id}==i));
                end
                N_dotj(i) = aux;
                % prod over all C_j
                prodC_j = (det(inv(params.invCj(:,:,i)))^0.5)*prodC_j;
                
            end
            % If any cluster is empty, remove and decrease J
            for iRem = 1:length(N_dotj)
                if N_dotj(iRem) == 0
                    fprintf('Removing class %d\n',iRem);
                    S = updateClusterAssignement_rem(S,params,iRem);
                    
                    params.J = params.J - 1;
                    numOfClasses = numOfClasses - 1;
                    params.invCj(:,:,iRem) = [];
                    params.mu_j(:,iRem) = [];
                    %                     params.ap = [];
                    %                     params.bp =[];
                end
            end
            % for S - eq 3 iwata is analog to variable c_i in
            % rasmussen99 cluster asignement S
            %             pS_gamma = ((params.gamma^numOfClasses)*prod(factorial(N_dotj-1)))/...
            %                 (params.gamma*prod(params.gamma+linspace(1,N-1,N-1)));
            % for probability of data X eq - 4 Iwata
            
            %             pX_SWabr = (2*pi)^(auxSumD)*params.r^((params.K*numOfClasses)/2)*...
            %                 (params.b^params.a)/(params.bp^params.ap)*(gamma(params.ap)/gamma(params.a))*...
            %                 prodC_j;
            %% Inference part II
            % sample sdn using probability p( j|X, S\dn,W, a, b, r, alpha ) (11)
            % from j = 1, . . . , J + 1
            %             s_dn (n) = randi(J); %prueba
                        
            prob = NaN*zeros(1,numOfClasses+1);
            prob_log = NaN*zeros(1,numOfClasses+1);
            probEq14 = NaN*zeros(1,numOfClasses+1);
            probEq14_log = NaN*zeros(1,numOfClasses+1);
            for j = 1:params.J
                [ap_sdn_j,bp_sdn_j,mu_j_sdn_j,invCj_sdn_j] = ...
                    equation13IWata_sdn_j(X,S,W,params,d,n,j);
                % Calculate p(s_dn=j ...) eq 14 for an existing cluster
                Nj_dn = Ndot_j_dn(S,d,n,j,params);
                prob_s_dn = Nj_dn/(N-1+params.gamma);
                %                 prob_s_dn_log = log(Nj_dn)-log(N-1+params.gamma);
                % Calculate eq 13 , p(X|s_dn=j ...)
                Sno_dn = S{d};
                Sno_dn(n) = [];
                Jeq13 = length(unique(Sno_dn));
                % likelihodd
                probEq14(j) = ((2*pi)^(-0.5*params.Md(d)))*(r^(0.5*(j>Jeq13)))*((params.bp_no_dn^params.ap_no_dn)/...
                    (bp_sdn_j(j)^ap_sdn_j))*(gamma(ap_sdn_j)/gamma(params.ap_no_dn))*...
                    ((det(inv(invCj_sdn_j(:,:,j)))^0.5)/(det(inv(params.invCj_no_dn(:,:,j)))^0.5));
                % Log of the likelihood
                probEq14_log(j) = ((2*pi)^(-0.5*params.Md(d)))*(r^(0.5*(j>Jeq13)))*exp((params.ap_no_dn*log(params.bp_no_dn))-...
                    (ap_sdn_j*log(bp_sdn_j(j))))*exp((gammaln(ap_sdn_j)-gammaln(params.ap_no_dn)))*...
                    ((det(inv(invCj_sdn_j(:,:,j)))^0.5)/(det(inv(params.invCj_no_dn(:,:,j)))^0.5));
                % Calculate eq 11, p(s_dn = j| X,S_dn,W,a,b,r,gamma)
                % revisar LOGDET gpmat
                %                 prob(j) = prob_s_dn*probEq14(j); % Eq 12 iwata
                prob(j) = prob_s_dn*probEq14_log(j); % Eq 12 iwata
            end
            % Now calculate the prior and likelihood for a new cluster
            xdn = X{d}(n,:);
%             probNew = probEq13NewCluster(params,W,X,d,n); %%%% voy AQUI
            [probNew,apNew,bpNew, mu_jNew, invCjNew] = probEq13NewCluster(params,W,X,d,n);

            % Eq 14 - iwata
            prob(end) = ((params.gamma)/(params.N-1+params.gamma))*probNew; % Eq 2 iwata
            
            % Sample a new value for s_dn from prob after normalizing
            probNorm = prob./sum(prob);
            % Since s_dn is draw from a categorical distribution
            s_dnNew = mnrnd(1,probNorm);
            % update according  to the new value of s_dn sampled and check
            % if a new cluster is started
            if  s_dnNew(end) == 1
                % Posterior for the precision parameter alpha and the laten
                % vector z_j                eq - 9 and 10 Iwatta
                params.J = (params.J)+1;
                S{d}(n) = params.J;
                params.S{d}(n) = params.J;
                % adds the new parameters
%                 params.ap = apNew;
%                 params.bp = bpNew;
                params.mu_j(:,params.J) = mu_jNew;
                params.invCj(:,:,params.J) = invCjNew;
%                 S = params.S;
%                 [ap,bp,mu_j,invCj] = equation4IWata(X,S,W,params);
                [bp] = equation4IWataUpdateNewCluster(X,S,W,params);
%                 params.ap = ap;
                params.bp = bp;
%                 params.mu_j = mu_j;
%                 params.invCj = invCj;
                
                pos_alpha = gamrnd(ap,1/bp);
                pos_zj(j,:) = mvnrnd(params.mu_j(:,j),(pos_alpha^-1)*inv(params.invCj(:,:,j)));
                
                numOfClasses          = numOfClasses + 1;
                fprintf('Adding new class %d\n',params.J);
            else
                [~ , S{d}(n)] = find(s_dnNew==1);
                
            end
            
            
        end
    end
    
    
% 
    %% M-step
    
    for d = 1:D
        if optiW == true
            %  update Wd using a numerical optimization method using (18)
            x0 = W{d}(:);
            out = ncg(@(x) projectionMatrixOptFCN(x,params,X,S,d),x0,'RelFuncTol',1e-16,'StopTol',1e-8,...
                'MaxFuncEvals',500,'Display','final');
            W{d} =reshape(out.X,params.Md(d),params.K);
        else
            %   update Wd using a exact equation (19)
            W{d} = MstepJointLLexact(X,S,params,d);
        end
    end
    
    ll(t) = log_likelihoodEq4(params);
    fprintf('Log-Likelihood: %f\n',ll(t));
end

figure,plot(ll,'-or','LineWidth',3,'MarkerFaceColor','b','MarkerSize',3);
xlabel('Iteration');
ylabel('Log-Likelihood');

%% Metrics 
Svect = [];
Struevect = [];
for d = 1:D
    Sd = S{d};
    Sdtrue = [trueS{d}]';
    Svect = [Svect;Sd];
    Struevect = [Struevect;Sdtrue];
    
end
[AR, Rand, Mirkin, Hubert] =  valid_RandIndex(Svect,Struevect);

% save  dataToyLINmodel4oct2D5J3K.mat S W X
% Posterior for the precision parameter alpha and the
% latent vector z_j    eq - 9 and 10 Iwatta
% pos_alpha = gamrnd(params.ap,params.bp);
% pos_zj(i,:) = mvnrnd(params.mu_j(:,i),(r^-1)*inv(params.invCj(:,:,i)));

