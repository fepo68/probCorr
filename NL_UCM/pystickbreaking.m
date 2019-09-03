function weights = pystickbreaking(alpha, sigma, K)

weights = zeros(K, 1);
weights(1) = betarnd(1, alpha + sigma);
for i=2:K
    b = betarnd(1-sigma, alpha + i*sigma);
    weights(i) = b * (1.0 - sum(weights(1:i)));
    if sum(weights)>=1.0        
        break;
    end
end