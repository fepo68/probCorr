function PHI_X = polyToyMapBasis(X,S,Md,Nd,K,fBasis,flag,kvar,bias)
PHI_X = {};
D = length(Md);
% fBasis = 'polynomial';
Mu1 = mean(X{1});
Mu2 = mean(X{2});
s = 0.01;
PHI_X = {};

if strcmp(fBasis,'poly')
    PHI_X = {};
    % if poly is set to on we use a poly basis function with degree = 3
    for d = 1:D
        Xd = X{d};
        phiX = zeros(Nd(d),4);
        for n = 1:Nd(d)
            xdn = Xd(n,:);
            %             m = 4
            phiX(n,:) = kvar*[xdn(1)^3,sqrt(3)*(xdn(1)^2)*xdn(2),sqrt(3)*xdn(1)*(xdn(2)^2),xdn(2)^3]';
            %             from a poly kernel of the form k(x,y) = (b+r*(x'*y))^3,
            %             m = 10;
            %             phiX(n,:) = [bias^3,sqrt(3)*bias*sqrt(kvar)*xdn(1),sqrt(3*bias)*kvar*xdn(1)^2,...
            %                 (sqrt(kvar)^3)*(xdn(1)^3), sqrt(3)*bias*sqrt(kvar)*xdn(2), sqrt(6*bias)*kvar*xdn(1)*xdn(2),...
            %                 sqrt(3)*(sqrt(kvar)^3)*(xdn(1)^2)*xdn(2),sqrt(3*bias)*kvar*xdn(2)^2,...
            %                  sqrt(3)*(sqrt(kvar)^3)*(xdn(2)^2)*xdn(1),(sqrt(kvar)^3)*(xdn(2)^3)];
        end
        if flag ~= true
            PHI_X{d} = phiX;
        else
            PHI_X{d} = zscore(phiX);
        end
    end
    
    
    return
end

if K == length(unique(S{1}))
    mu_k = {};
    s2 = {};
    for d = 1:D
        s2{d} = 20*mean(mean(pdist2(X{d},X{d})));
        aux_mu_k = zeros(K,Md(d));
        for k = 1:K
            [~,idk] = find(S{d}==k);
            %             mu_k(k,:) = mean(X{d}(idk,:));
            aux_mu_k(k,:) = mean(X{d}(idk,:));
        end
        mu_k{d} = aux_mu_k;
        
    end
end

if K > length(unique(S{1}))
    disp('Initialize the mu_i of the Basis Functions with K-Means')
    s2 = {};
    mu_k = {};
    for d = 1:D
        % Set up cluster model
        ncentres = K;
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


for d = 1:D
    mu_kd = mu_k{d};
    phiX = [];
    for k = 1:size(mu_kd,1)
        if strcmp(fBasis,'gauss')
            phiX(:,k) = exp(-(1/(2*s2{d}))*diag((X{d}-...
                (repmat(mu_kd(k,:),Nd(d),1)))*(X{d}-...
                repmat(mu_kd(k,:),Nd(d),1))'));
        elseif strcmp(fBasis,'sigmoid')
            %             phiX(:,k) = 1./(exp(-(1/(sqrt(s2{d})))*diag((X{d}-...
            %                 (repmat(mu_kd(k,:),Nd(d),1)))*(X{d}-...
            %                 repmat(mu_kd(k,:),Nd(d),1))'))+1);
            phiX(:,k) = tanh(kvar*diag(X{d}*repmat(mu_kd(k,:),Nd(d),1)')+bias);
%             phiX(:,k) = tansig((1/(100*sqrt(s2{d})))*diag((X{d}-...
%                (repmat(mu_kd(k,:),Nd(d),1)))*(X{d}-...
%                 repmat(mu_kd(k,:),Nd(d),1))'));
            
        end
    end
    
    if flag ~= true
        PHI_X{d} = phiX;
    else
        PHI_X{d} = zscore(phiX);
    end
    
end






% for d = 1:D
%     Xd = X{d};
%     [Nd,Md] = size(Xd);
%     %     phiX = zeros(Nd(d),kern.degree);
%     phiX =[];
%     for n = 1:Nd
%         x = Xd(n,:);
%
%
%
%         if strcmp(fBasis,'polynomial')
%             xmap = designMatrix(x,'polynomial',kern,kern.degree);
%             %             m = 0:kern.degree-1;
%             %             xmap = kern.variance*(x*x'*kern.weightVariance+kern.biasVariance).^m;
%         elseif strcmp(fBasis,'sigmoid')
%
%             xmap = designMatrix(x,'sigmoid',Mu1,Mu2,s);
%         end
%
%
%         %         phiX(n,:) = xmap;
%         phiX =[phiX;xmap];
%     end
%
%     PHI_X{d} = phiX;
% end