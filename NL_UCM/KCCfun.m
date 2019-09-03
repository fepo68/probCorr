function logP = KCCfun(sigma2, varargin)
    XTrain = varargin{1};
    yTrain = varargin{2};
    hmmXTrain = varargin{3};
    distype = 'KE';
    sigma2 = exp(sigma2);
     
    N = length(XTrain);
    H = eye(N) - ones(N)/N;
    L = repmat(yTrain,1,length(yTrain)) - repmat(yTrain',length(yTrain),1) + 1;
    L(L~=1) = 0;
%     K = hmm_based_proximity_analysis(XTrain, XTrain, hmmXTrain, hmmXTrain, distype, sigma2, true);
    K = hmm_based_proximity_analysis(XTrain, [], hmmXTrain, [], distype, sigma2, true);
    KH = K*H;
    LH = L*H; 
      
    logP = log(trace(LH*KH)) - log(sqrt(trace(KH*KH))) - log(sqrt(trace(LH*LH)));
    logP = -logP;
    
end