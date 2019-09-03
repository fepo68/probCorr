function [S,W,X,PHIX] = createToyLVMClusterBasisFunc(Nd,Md,D,J,K,fBasis,nfBasis)
% First, we sampled latent vectors zj for j=1,...,J? from a
% K-dimensional normal distribution with mean 0 and covariance I.
Z = zeros(K,J);
for j = 1:J
    Z(:,j) = mvnrnd(zeros(K,1),eye(K));
end

% Next, we generated projection matrices Wd for d=1,2, ...
% where each element is drawn from a normal distribution with mean 0 and variance 1.
W  = {};
for d = 1:D
    Wd = normrnd(0,1,[Md(d),K]);
    W{d} = Wd;
end

% Finally, we generated N/J? objects for each cluster j
% using a normal distribution with mean Wd*zj and covariance (alpha^-1)*I,
% and obtained N objects in total for each domain d=1,2 ...
% 2. Draw a precision parameter
% with a = 1 and b = 1
alpha = gamrnd(1,1);
Nd_j = (sum(Nd)/J)/D;
X = {};
S = {};
for d = 1:D
    Wd = W{d};
    %     Xd = zeros(Md(d),Nd(d));
    Xd = [];
    Sd = [];
    
    savedState = {};
    for j = 1:J
        for n = 1:Nd_j
            stream = RandStream.getGlobalStream;
            savedState{j,n} = stream.State;
            
            x_dn = mvnrnd(Wd*Z(:,j),((alpha)^(-1))*eye(Md(d)));
            Xd = [Xd,x_dn'];
            Sd = [Sd,j];
        end
    end
    X{d} = Xd';
    S{d} = Sd;
    
end


if nfBasis(d) == length(unique(S{1}))
    mu_k = {};
    s2 = {};
    for d = 1:D
        s2{d}= mean(mean(pdist2(X{d},X{d})));
        aux_mu_k = zeros(nCluster,Md(d));
        for k = 1:K
            [~,idk] = find(S{d}==k);
            %             mu_k(k,:) = mean(X{d}(idk,:));
            aux_mu_k(k,:) = mean(X{d}(idk,:));
        end
        mu_k{d} = aux_mu_k;
        
    end
end

if nfBasis(d) > length(unique(S{1}))
    disp('Initialize the mu_i of the Basis Functions with K-Means')
    s2 = {};
    mu_k = {};
    for d = 1:D
        % Set up cluster model
        ncentres = nfBasis(d);
        centres = zeros(ncentres, Md(d));
        
        % Set up vector of options for kmeans trainer
        options = foptions;
        options(1)  = 1;		% Prints out error values.
        options(5) = 1;
        options(14) = 10;		% Number of iterations.
        
        clc
        disp('The model is chosen to have K centres, which are initialised')
        disp('at randomly selected data points.  We now train the model using')
        disp('the batch K-means algorithm with a maximum of 10 iterations and')
        disp('stopping tolerance of 1e-4.')
        disp(' ')
        
        % Train the centres from the data
        [aux_mu_k, options, post] = kmeans(centres, X{d}, options);
        mu_k{d} = aux_mu_k;
        s2{d} =  mean(mean(pdist2(aux_mu_k,aux_mu_k)));
    end
end

%% Restore the Stream and compute the mapping Features

PHIX = {};
S = {};
for d = 1:D
    mu_kd = mu_k{d};
    Wd = W{d};
    %     Xd = zeros(Md(d),Nd(d));
    Xd = [];
    Sd = [];
    
%     savedState = {};
    for j = 1:J
        for n = 1:Nd_j
            stream.State = savedState{j,n};
            aux_xdn = Wd*Z(:,j);
            val = zeros(size(mu_kd,1),1);
            for k = 1:size(mu_kd,1)
                if strcmp(fBasis,'gauss')
                    val(k,1) = exp(-(1/(2*s2{d}))*(aux_xdn'-mu_kd(k,:))*(aux_xdn'-mu_kd(k,:))');
%                 elseif strcmp(fBasis,'sigmoid')
%                     phiX(:,k) = 1./(exp(-(1/(sqrt(s2{d})))*diag((X{d}-...
%                         (repmat(mu_kd(k,:),Nd(d),1)))*(X{d}-...
%                         repmat(mu_kd(k,:),Nd(d),1))'))+1);
                    
                end
            end
            
            x_dn = mvnrnd(val,((alpha)^(-1))*eye(nfBasis(d)));
            Xd = [Xd,x_dn'];
            Sd = [Sd,j];
        end
    end
    PHIX{d} = Xd';
    S{d} = Sd;
    
end


% save(['dataToyIwatamodel03Feb16_5',num2str(D),'D',num2str(Nd_j),'Ndj',num2str(J),'J',num2str(K),'K.mat'],'S','W','X','Nd','Md','D','J','K');
