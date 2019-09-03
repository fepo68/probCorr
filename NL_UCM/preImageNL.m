function [f] =  preImageNL(x,ydn,degree,kvar)

% By using a plynomial basis function with degree = 2 and dimension of x
% \in R^2
if degree == 2
    phi_xdn = [x(1)^2,sqrt(2)*prod(x),x(2)^2]';
elseif degree ==3
    phi_xdn = kvar*[x(1)^3,sqrt(3)*x(1)^2 *x(2),sqrt(3)*x(1)*x(2)^2,x(2)^3]';
end

f = sum((ydn-phi_xdn).^2);