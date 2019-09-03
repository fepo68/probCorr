%% Genera Toy PhD. UCM-LVM, as in Iwata's Paper from 2016

clear all
close all
clc

J = 5;
D = 2;
K = 3;
Md = [5 5];
Nd = [200 200];

for D = 2:2 % for several domains experiments
    
    Md = Md(D)*ones(1,D);
    Nd = 200*ones(1,D);
    
    for i = 1:5
        
%         seed = 1e5*3;
        s = RandStream('mt19937ar','Seed',1e5*i);
        RandStream.setGlobalStream(s);
        
        linToy = true;
        if linToy == true
            
            [S,W,X] = createToyLVMCluster(Nd,Md,D,J,K); % For the linear
            %         model
            plot(X{1}(:,1),X{1}(:,2),'*r')
            
            save(['toyData/synth',num2str(K),'exp',num2str(i),'D',num2str(D),'26feb18.mat'],'S','W','X','Nd','Md','D','J','K');
        else
            
            [S,W,X,PHIX] = createToyLVMClusterBasisFunc(Nd,Md,D,J,K,'gauss',[20 20]);
            plot(X{1}(:,1),X{1}(:,2),'*r')
            
            save(['synth',num2str(K),'exp',num2str(i),'D',num2str(D),'Md5_5.mat'],'S','W','X','Nd','Md','D','J','K','PHIX');
        end
        
        % for the nonlinear model
        
        
    end
end