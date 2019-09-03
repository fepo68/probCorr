function [S,W,X,Xinput] = createToyLVMCluster_NonLinearMaapping(Nd,Md,D,J,K)

% s = RandStream('mt19937ar','Seed',seed);
% RandStream.setGlobalStream(s);
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
% Xinput = {};
S = {};
for d = 1:D
    Wd = W{d};
    %     Xd = zeros(Md(d),Nd(d));
    Xd = [];
    Sd = [];
    %     Xdinput = [];
    for j = 1:J
        for n = 1:Nd_j
            x_dn = mvnrnd(Wd*Z(:,j),((alpha)^(-1))*eye(Md(d)));
            Xd = [Xd,x_dn'];
            Sd = [Sd,j];
            
            % At this point, X is referred to the features in the Hilbert space
            
            % We compute the observed features (Input space) from
            
            % x_dn* = arg_min ||Wd*z_{s_dn} - \phi(x_dn)||_2^2
            
            % We set the basis function parameters such as
            
            % We use a poly basis function of degree (d = 2 default)
            
            % let's test with a given Wd*z_j
            %             d = 1;
            %             j = 1;
            %             y_dn = W{d}*Z(:,j);
            %             y_dn = x_dn';
            %             degree = 3;
            %             kvar = 1;
            %
            %             %             fval = inf;
            %             fval = zeros(1,5);
            %             x_optC = {};
            %             for fvalid = 1:20
            %                 x0 = randn(2,1);
            %                 fun = @(x) preImageNL(x,y_dn,degree,kvar); % Without Gradients
            %                 %                 opt = optimset('GradObj','on'); % This is how to specify options for fminunc
            %                 %                 opt = optimset(opt,'TolX',1e-6);
            %                 % opt = optimset('TolX',1e-12);
            %                 %                 opt = optimset(opt,'LargeScale','off');
            %                 %                 opt = optimset(opt,'Display','iter');
            %                 % opt = optimset(opt,'Algorithm','levenberg-marquardt');
            % %                 opt = optimoptions(@fminunc,'Display','iter','Algorithm','trust-region',...
            % %                     'SpecifyObjectiveGradient','true','TolFun',1e-12,'TolX',1e-12);
            %                 opt = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','TolFun',1e-12,'TolX',1e-12);
            %                 try
            %                     [x_opt,fval(fvalid),~,~,grad] = fminunc(fun,x0,opt);
            %                 catch
            %                     x0 = randn(2,1);
            %                     [x_opt,fval(fvalid),~,~,grad] = fminunc(fun,x0,opt);
            %                 end
            %                 fprintf('Gradient: %f\n',norm(grad));
            %                 x_optC{fvalid} = x_opt;
            %             end
            %             [~,idmin] = min(fval);
            %             x_opt = x_optC{idmin};
            %             Xdinput = [Xdinput,x_opt];
        end
    end
    X{d} = Xd';
    %     Xinput{d} = Xdinput';
    S{d} = Sd;
    
end


% Preimage compute for observed x_dn
Xinput = {};
for d = 1:D
    Wd = W{d};
    %     Xd = zeros(Md(d),Nd(d));
    Xdinput = [];
    %% auto parameter for lengthscale
    DIST=distanceMatrix(X{d});
    DIST(DIST==0)=inf;
    DIST=min(DIST);
    para=5*mean(DIST);
    [Y, eigVector, eigValue]= mykPCA(X{d},4,'gaussian',para);
    %     for j = 1:J
    for n = 1:Nd(d)
        %             Phi_xdn
        
        %% At this point, X is referred to the features in the Hilbert space
        
        % We compute the observed features (Input space) from
        
        % x_dn* = arg_min ||Wd*z_{s_dn} - \phi(x_dn)||_2^2
        
        % We set the basis function parameters such as
        
        % We use a poly basis function of degree (d = 2 default)
        
        % let's test with a given Wd*z_j
        %             y_dn = X{d}(n,:)';
        y_dn = Y(n,:)';
        degree = 3;
        kvar = 1;
        
        fval = inf;
        fval = zeros(1,5);
        x_optC = {};
%         x0 = Y(n,:);
        for fvalid = 1:20
            x0 = randn(2,1);
            fun = @(x) preImageNL(x,y_dn,degree,kvar); % Without Gradients
            opt = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','TolFun',1e-12,'TolX',1e-12);
            try
                [x_opt,fval(fvalid),~,~,grad] = fminunc(fun,x0,opt);
            catch
                x0 = randn(2,1);
                [x_opt,fval(fvalid),~,~,grad] = fminunc(fun,x0,opt);
            end
            fprintf('Gradient: %f\n',norm(grad));
            x_optC{fvalid} = x_opt;
        end
        [~,idmin] = min(fval);
        x_opt = x_optC{idmin};
        Xdinput = [Xdinput,x_opt];
        
        % %% kPCA data processing
        % meanY=mean(Y)';
        % stdY=std(Y)';
        % z=mykPCA_PreImage(meanY,eigVector,Y,para);
        
    end
    %     end
%     X{d} = Xd';
%     Xinput{d} = Xdinput';
%     S{d} = Sd;
    
end



% At this point, X is referred to the features in the Hilbert space

% We compute the observed features (Input space) from

% x_dn* = arg_min ||Wd*z_{s_dn} - \phi(x_dn)||_2^2

% We set the basis function parameters such as

% We use a poly basis function of degree (d = 2 default)

% let's test with a given Wd*z_j
% d = 1;
% j = 1;
% y_dn = W{d}*Z(:,j);
% degree = 3;
%
% x0 = rand(2,1);
% fun = @(x) preImageNL(x,y_dn,degree); % Without Gradients
% %             opt = optimset('GradObj','on'); % This is how to specify options for fminunc
% %             opt = optimset(opt,'TolX',1e-6);
% % opt = optimset('TolX',1e-12);
% % % opt = optimset(opt,'LargeScale','off');
% % opt = optimset(opt,'Display','iter');
% % opt = optimset(opt,'Algorithm','levenberg-marquardt');
% opt = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','TolFun',1e-12,'TolX',1e-12);
% [x_opt,~,~,~,grad] = fminunc(fun,x0,opt);
% fprintf('Gradient: %f\n',norm(grad));

% save(['dataToyIwatamodel03Feb16_5',num2str(D),'D',num2str(Nd_j),'Ndj',num2str(J),'J',num2str(K),'K.mat'],'S','W','X','Nd','Md','D','J','K');
