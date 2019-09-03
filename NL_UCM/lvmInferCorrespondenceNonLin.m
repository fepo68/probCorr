function [X,S,W,params] = lvmInferCorrespondenceNonLin(T,D,K,alphaW,X)
% For the inference of the nonlinear model

% T: number of iterations
% D: number of domains
% K: dimensionality of the latent vector
% X: Observed data

% model parameters
a = 1;
b = 1;
r = 1;
gammaVal = 1;
% K = 3;



optiW = true;

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
%     cs{d} = S{d};
%     numOfPointsInClass = hist(cs{d},1:numOfClasses);
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

%% Kernel Parameters
params.kernType = 'linear';
% Since we have three different Kernels
% for Wd'Wd Kernel
params.varianceKww = 0.1;
val = mean(mean(dist2(W{1}',W{1}')));
params.inversewithKww = 1/(val)^2;
% for Wd'xdn Kernel
params.varianceKwx = 0.1;
val = mean(mean(dist2(W{1}',X{1}(1,:))));
params.inversewithKwx = 1/(val)^2;
% for xdn'xdn Kernel
params.varianceKxx = 0.1;
val = mean(mean(dist2(X{1}(1,:),X{2}(1,:))));
params.inversewithKxx = 1/(val)^2;


%% Second factor parameters depicted in (4)
[ap,bp,mu_j,invCj] = equation4IWataNonLinear(X,S,W,params);
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
                equation13IWata_excluding_dnNonLinear(X,S,W,params,d,n);
            
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
                    equation13IWata_sdn_jNonLinear(X,S,W,params,d,n,j);
                % Calculate p(s_dn=j ...) eq 14 for an existing cluster
                Nj_dn = Ndot_j_dn(S,d,n,j,params);
                prob_s_dn = Nj_dn/(N-1+params.gamma);
                %                 prob_s_dn_log = log(Nj_dn)-log(N-1+params.gamma);
                % Calculate eq 13 , p(X|s_dn=j ...)
                Sno_dn = S{d};
                Sno_dn(n) = [];
                Jeq13 = length(unique(Sno_dn));
                % likelihodd
                %                 probEq14(j) = ((2*pi)^(-0.5*params.Md(d)))*(r^(0.5*(j>Jeq13)))*((params.bp_no_dn^params.ap_no_dn)/...
                %                     (bp_sdn_j(j)^ap_sdn_j))*(gamma(ap_sdn_j)/gamma(params.ap_no_dn))*...
                %                     ((det(inv(invCj_sdn_j(:,:,j)))^0.5)/(det(inv(params.invCj_no_dn(:,:,j)))^0.5));
                % Log of the likelihood
                probEq14_log(j) = ((2*pi)^(-0.5*params.Md(d)))*exp((params.ap_no_dn*log(params.bp_no_dn))-...
                    (ap_sdn_j*log(bp_sdn_j(j))))*exp((gammaln(ap_sdn_j)-gammaln(params.ap_no_dn)))*...
                    (((1/det(invCj_sdn_j(:,:,j)))^0.5)/((1/det(params.invCj_no_dn(:,:,j)))^0.5));
                % Calculate eq 11, p(s_dn = j| X,S_dn,W,a,b,r,gamma)
                % revisar LOGDET gpmat
                %                 prob(j) = prob_s_dn*probEq14(j); % Eq 12 iwata
                prob(j) = prob_s_dn*probEq14_log(j); % Eq 12 iwata
            end
            % Now calculate the prior and likelihood for a new cluster
            xdn = X{d}(n,:);
            %             probNew = probEq13NewCluster(params,W,X,d,n); %
            [probNew,apNew,bpNew, mu_jNew, invCjNew] = probEq13NewClusterCDGNonLinear(params,W,X,d,n);
            
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
                [bp] = equation4IWataUpdateNewClusterNonLinear(X,S,W,params);
                %                 params.ap = ap;
                params.bp = bp;
                %                 params.mu_j = mu_j;
                %                 params.invCj = invCj;
                
                %                 pos_alpha = gamrnd(ap,1/bp);
                %                 pos_zj(j,:) = mvnrnd(params.mu_j(:,j),(pos_alpha^-1)*inv(params.invCj(:,:,j)));
                
                numOfClasses          = numOfClasses + 1;
                fprintf('Adding new class %d\n',params.J);
            else
                
                try
                    [~ , S{d}(n)] = find(s_dnNew==1);
                catch
                    S{d}(n) = 1;
                end
                
            end
            
            %% Search for empty clusters
            %     prodC_j = 1;
            N_dotj =[];
            for i = 1:numOfClasses
                %         objects assigned to cluster j, Iwata
                aux = 0;
                for id = 1:D
                    aux = aux + length(find(S{id}==i));
                end
                N_dotj(i) = aux;
                %         prod over all C_j
                %         prodC_j = (det(inv(params.invCj(:,:,i)))^0.5)*prodC_j;
                
            end
            %     If any cluster is empty, remove and decrease J
            for iRem = 1:length(N_dotj)
                if N_dotj(iRem) == 0
                    fprintf('Removing class %d\n',iRem);
                    S = updateClusterAssignement_rem(S,params,iRem);
                    
                    params.J = params.J - 1;
                    numOfClasses = numOfClasses - 1;
                    params.invCj(:,:,iRem) = [];
                    params.mu_j(:,iRem) = [];
                    params.invCj_no_dn(:,:,iRem) = [];
                    params.mu_j_no_dn(:,iRem) = [];
                    [bp] = equation4IWataUpdateNewClusterNonLinear(X,S,W,params);
                    %             params.ap = [];
                    %              params.bp =[];
                end
            end
        end
    end
    
    
    %
    %% M-step
    
    for d = 1:D
        if optiW == true
            
            %  update Wd using a numerical optimization method using (18)
            if ~strcmp(params.kernType,'linear')
                x0 = [];
                x0(1) = params.varianceKxx;
                x0(2) = params.inversewithKxx;
                x0(3) = params.varianceKwx;
                x0(4) = params.inversewithKwx;
                x0(5) = params.varianceKww;
                x0(6) = params.inversewithKww;
                x0 = [x0,W{d}(:)'];
                fun = @(x) likelihood_WGradOffNonLin(x,params,W,X,S,d); % Without Gradients
                
                %             opt = optimset('GradObj','on'); % This is how to specify options for fminunc
                opt = optimset('TolX',1e-6);
                opt = optimset(opt,'LargeScale','off');
                opt = optimset(opt,'Display','iter');
                
                
                %             [w_opt,fval,exitflag,output,grad] = fminunc(fun,x0,opt);
                lb = [1e-6,1e-6,1e-6,1e-15,1e-6,1e-6,-Inf*ones(1,length(x0(7:end)))];
                ub = [Inf,Inf,Inf,Inf,Inf,Inf,Inf*ones(1,length(x0(7:end)))];
                [w_opt,~,~,~,~,grad,~] = fmincon(fun,x0,[],[],[],[],lb,ub,[],opt);
                
                fprintf('Gradient: %f\n',norm(grad));
                
                params.varianceKxx = w_opt(1);
                params.inversewithKxx = w_opt(2);
                
                params.varianceKwx = w_opt(3);
                params.inversewithKwx = w_opt(4);
                
                params.varianceKww = w_opt(5);
                params.inversewithKww = w_opt(6);
                W{d} =reshape(w_opt(7:end)',params.Md(d),params.K);
            else
                x0 = W{d}(:)';
                fun = @(x) likelihood_WGradOffNonLin(x,params,W,X,S,d); % Without Gradients
                
                opt = optimset('GradObj','on'); % This is how to specify options for fminunc
                opt = optimset(opt,'TolX',1e-6);
                opt = optimset(opt,'LargeScale','off');
                opt = optimset(opt,'Display','iter');
                [w_opt,fval,exitflag,output,grad] = fminunc(fun,x0,opt);
                fprintf('Gradient: %f\n',norm(grad));
                 W{d} =reshape(w_opt',params.Md(d),params.K);
            end
            
            
            
        else
            %   update Wd using a exact equation (19)
            W{d} = MstepJointLLexact(X,S,params,d);
        end
    end
    
    %% Second factor parameters depicted in (4)
    [ap,bp,mu_j,invCj] = equation4IWataNonLinear(X,S,W,params);
    params.ap = ap;
    params.bp = bp;
    params.mu_j = mu_j;
    params.invCj = invCj;
    
    
    
    
    ll(t) = log_likelihoodEq4(S,params);
    fprintf('Log-Likelihood: %f\n',ll(t));
end

figure,plot(ll,'-or','LineWidth',3,'MarkerFaceColor','b','MarkerSize',3);
xlabel('Iteration');
ylabel('Log-Likelihood');



% save  dataToyLINmodel4oct2D5J3K.mat S W X
% Posterior for the precision parameter alpha and the
% latent vector z_j    eq - 9 and 10 Iwatta
% pos_alpha = gamrnd(params.ap,params.bp);
% pos_zj(i,:) = mvnrnd(params.mu_j(:,i),(r^-1)*inv(params.invCj(:,:,i)));

