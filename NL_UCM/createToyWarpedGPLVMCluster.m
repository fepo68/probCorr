function [S,X,Fd] = createToyWarpedGPLVMCluster(Nd,Md,D,J,K)

% s = RandStream('mt19937ar','Seed',seed);
% RandStream.setGlobalStream(s);
% First, for each cluster c = 1 , ... , infty
alpha = gamrnd(1,1);
Rc = zeros(K,K,J);
muc = zeros(K,J);
invS = eye(K);
v = K*2;
r = 1;
for j = 1:J
    % Draw precision
    Rc(:,:,j) = wishrnd(invS,v)/v;
    muc(:,j) = mvnrnd(zeros(K,1),((r*Rc(:,:,j))\eye(K)));
end

%% Draw latent coordinates and latent functions
Z = {}; % latent assignment
X = {}; % latent corrdinates
Y = {}; % input data (observation)
lambda = ones(1,J)*(1/J); % probabiliti weights
for d = 1:D
    Z{d} = mnrnd(1,lambda,Nd(d));
    Xd = zeros(Nd(d),K);
    for n = 1:Nd(d)
        Xd(n,:) = mvnrnd(muZn,invRzn);
    end 
end


%%%%%%%%%%%%%%%%55
% Finally, we generated N/J? objects for each cluster j
% using a normal distribution with mean fdm(zj) and covariance (alpha^-1)*I,
% and obtained N objects in total for each domain d=1,2 ...
% 2. Draw a precision parameter
% with a = 1 and b = 1

Nd_j = (sum(Nd)/J)/D;
X = {};
S = {};
Fd = {};
for d = 1:D
    %     Xd = zeros(Md(d),Nd(d));
    Xd = [];
    Sd = [];
    Sd = mnrnd(1,ones(1,J)*(1/J),Nd(d));
    Zd = Sd*Z';
    
    % Build the kernel for z
    %     z = Z';
    z = Zd;
    kern = kernCreate(z, 'rbf');
    kern.inverseWidth = 0.1;
    kern.variance = 1;
    %         figure(1)
        clf
    K = kernCompute(kern, z);
        imagesc(K);
    % need to take the real part of the sample as the kernel is numerically less than full rank
    % Sample fdm using a fdm(z)~GP(0,K)
    Fd{d} = real(gsamp(zeros(1, size(z, 1)), K, Md(d)))';
    Xd = [];
    %     Sd = [];
    %     for j = 1:J
    for n = 1:length(Fd{d})
        x_dn = mvnrnd(Fd{d}(n,:),((alpha)^(-1))*eye(Md(d)));
        Xd = [Xd;x_dn];
        %             Sd = [Sd,j];
    end
    %     end
    X{d} = Xd;
    S{d} = Sd;
    
end



% save(['dataToyIwatamodel03Feb16_5',num2str(D),'D',num2str(Nd_j),'Ndj',num2str(J),'J',num2str(K),'K.mat'],'S','W','X','Nd','Md','D','J','K');
