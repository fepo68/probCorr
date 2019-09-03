function dlogP_dS = KCCgrad(sigma2, varargin)
    XTrain = varargin{1};
    yTrain = varargin{2};
    hmmXTrain = varargin{3};
    distype = 'KE';
    sigma2 = exp(sigma2);    
     
    N = length(XTrain);
    H = eye(N) - ones(N)/N;
    L = repmat(yTrain,1,length(yTrain)) - repmat(yTrain',length(yTrain),1) + 1;
    L(L~=1) = 0;
    K = hmm_based_proximity_analysis(XTrain, XTrain, hmmXTrain, hmmXTrain, distype, sigma2, true);
% 	K2 = hmm_based_proximity_analysis(XTrain, [], hmmXTrain, [], distype, sigma2, true);
    KH = K*H;
    LH = L*H; 
    
	dgamma2_dS = hmm_based_proximity_gradient(XTrain, XTrain, hmmXTrain, hmmXTrain, sigma2);
% 	dgamma2_dS2 = hmm_based_proximity_gradient(XTrain, [], hmmXTrain, [], sigma2);
    
    dK_dS = -K.*dgamma2_dS;
    dK_dS = dK_dS*sigma2;
%     dK_dS = 2*dK_dS*sigma; % derivative of S w.r.t. sigma (i.e. sigma^2)
%  	dK_dS = dK_dS*exp(sigma2); % derivative of S w.r.t. sigma (i.e. exp(sigma^2))
    
    dlogP_dK = H*LH/trace(LH*KH) - H*KH/trace(KH*KH);
    dlogP_dS = -trace(dlogP_dK'*dK_dS);

end