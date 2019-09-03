function [S,W,X] = createToyLVMCluster(Nd,Md,D,J,K)

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
S = {};
for d = 1:D
    Wd = W{d};
    %     Xd = zeros(Md(d),Nd(d));
    Xd = [];
    Sd = [];
    for j = 1:J
        for n = 1:Nd_j
            x_dn = mvnrnd(Wd*Z(:,j),((alpha)^(-1))*eye(Md(d)));
            Xd = [Xd,x_dn'];
            Sd = [Sd,j];
        end
    end
    X{d} = Xd';
    S{d} = Sd;
    
end



% save(['dataToyIwatamodel03Feb16_5',num2str(D),'D',num2str(Nd_j),'Ndj',num2str(J),'J',num2str(K),'K.mat'],'S','W','X','Nd','Md','D','J','K');
