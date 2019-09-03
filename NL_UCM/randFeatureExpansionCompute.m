%% Random Fourier Features
% Random feature expansion for Latent Variable Models
function PHIX = randFeatureExpansionCompute(X,Nd,Md,Nrf,flag,approach)
D = length(Nd);
PHIX = {};
for d = 1:D
    
    if flag ~= true
        Xd = X{d};
    else
        Xd = zscore(X{d});
    end
    
    if approach == 1 %Cutajar17
        phiX = zeros(Nd(d),Md(d)*Nrf);
        mu = zeros(Md(d),1);
        Sigma = eye(Md(d));
        for n = 1:Nd(d)
            xdn = Xd(n,:)';
            auxPhi = [];
            for i = 1:Nrf
                w_rf = mvnrnd(mu,Sigma)';
                auxPhi = [auxPhi,cos(xdn'*w_rf),sin(xdn'*w_rf)];
            end
            phiX(n,:) = auxPhi;
            
        end
    elseif approach == 2 %Ali-Rahimi07
        gamma = 1e-2;
        regParam = 1e-6;
        
        NTrain = size(Xd',2);
        D = size(Xd',1);
        NProj = Nrf;
        
        W = sqrt(2*gamma)*randn(NProj,D);
        bias = 2*pi*rand(NProj,1);
        
        phiX = cos(W*Xd'+repmat(bias,1,size(Xd',2)))/sqrt(NProj);
        
        
    end
    PHIX{d} = phiX';
end