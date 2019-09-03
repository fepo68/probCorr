%% Unsupervised clustering matching via Probabilistic Latent Variable model
clear all
close all
clc
% Implementacion del paper de IWATA
% Set the random seed.
addpath(genpath('E:\Clases Doctorado\Applied Bayesian Nonparametrics\GPmat-master\'));
addpath(genpath('C:\Users\digital065\Dropbox\Pc_pereira\Pruebas_Nuevas\netlab'));
seed = 0;
randn('state', seed);
rand('twister', seed);
% D Number of domains
D = 2;
% Nd Number of objects in the dth domain
Nd = [100,100];
% Md Dimensionality of observed features in the dth domain
Md = [5,5];
% K Dimensionality of a latent vector
K = 3;
% J Number of clusters (latent vectors)to which objects are assigned
J = 5;
% xdn Observation of the nth object in the dth domain, xdn ? R->Md
% zj Latent vector for the jth cluster, zj ? R->K
% Wd Projection matrix for the dth domain, Wd ? R->Md x K
% Theta_j Mixture weight for the jth cluster,
a = 1;
b = 1;
r = 1;
gammaVal = 1;
% 1. Draw mixture weights Stick breaking process

theta = dpstickbreaking(gammaVal,J);
theta(end) = theta(end)+(1-sum(theta));
bar(1:length(theta),theta);

% 2. Draw a precision parameter
alpha = gamrnd(1,1);

% 3. For each cluster: j = 1, . . . , inf
% (a)Draw a latent vector zj?N(0,(?r)^{-1} *I)
Z = zeros(J,K);
for j=1:J
    Z(j,:) = mvnrnd(zeros(K,1),((alpha*r)^(-1))*eye(K));
end
% 4. For each domain: d = 1, ... , D
% (a) For each object: n=1,...,Nd
% i. Draw a cluster assignment sdn?Categorical(?)
% ii.Draw an observation vector xdn?N(Wd*z_sdn,??1*I)
X = {};
S = {};
linModel = true;
for d = 1:D
    Wd = normrnd(0,1,[Md(d),K]);
    Xd = zeros(Nd(d),Md(d));
    Sd = zeros(Nd(d),J);
    for n = 1:Nd(d)
        s_dn = mnrnd(1,theta);
        
        if linModel == true
            x_dn = mvnrnd(Wd*Z(logical(s_dn),:)',((alpha)^(-1))*eye(Md(d)));
            
            %%%%%
        else
            kern = kernCreate(Wd*Z(find(s_dn),:)','rbf');
            k = kernCompute(kern,Wd*Z(find(s_dn),:)');
            mu = tanh(Wd*Z(find(s_dn),:)');
            x_dn = gsamp(mu,k,1);
        end
        %%%%%
        
        Xd(n,:) = x_dn;
        Sd(n,:) = s_dn;
    end
    X{d} = Xd;
    S{d} = Sd;
    W{d} = Wd;
end

% S1 = X{1};
% S2 = X{2};


% if Md(1) == 2 && Md(2) == 2
%     plot(S1(:,1),S1(:,2),'dr','MarkerSize',5);
%     hold on
%     plot(S2(:,1),S2(:,2),'ob','MarkerSize',5);
% else
%     plot3(S1(:,1),S1(:,2),S1(:,3),'dr');
%     hold on
%     plot3(S2(:,1),S2(:,2),S2(:,3),'ob');
% end

save(['dataToyLINmodel20nov',num2str(D),'D',num2str(J),'J',num2str(K),'K.mat'],'S','W','X','Nd','Md','D','J','K');
% figure
% cont = 1;
% for j=1:D
%     % a. for each object: n=1, ...,Nd
%     for k=1:Nd
%         % i. Draw a cluster assignment
%         s_dn = mnrnd(1,theta',1);
%         subplot(D,Nd,cont),plot(s_dn,'dr','MarkerFaceColor','y','MarkerSize',1.2);
%         cont = cont+1;
%     end
% end
