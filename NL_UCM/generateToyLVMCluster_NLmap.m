%% Genera Toy PhD. UCM-LVM, as in Iwata's Paper from 2016

clear all
close all
clc

J = 5;
D = 2;
K = 3;
Md = [4 4];
Nd = [200 200];
pathData = './toyData/';
for D = 2:2 % for several domains experiments
    
    Md = Md(D)*ones(1,D);
    Nd = 200*ones(1,D);
    
    for i = 1:5
        
        %         seed = 1e5*3;
        s = RandStream('mt19937ar','Seed',1e5*i);
        RandStream.setGlobalStream(s);
        [S,W,X,Xinput] = createToyLVMCluster_NonLinearMaapping(Nd,Md,D,J,K); % For the nonlinear
        %         model
%         plot(X{1}(:,1),X{1}(:,2),'*r')
        
        save([pathData,'synth',num2str(K),'exp',num2str(i),'D',num2str(D),'J',num2str(J),'NLmap23oct17.mat'],'S','W','X','Nd','Md','D','J','K');
    end
    
end