function Wnew = MstepJointLLexact(X,S,params,d)

Saux = S{d};
Xd = X{d};

auxLeft = zeros(params.K);
auxRigth = zeros(length(Xd(1,:)),params.K);
for j = 1:params.J
    % Fin the number of objects in the domain d assigned to the cluster j
    [posObj,nObj_j] = find(Saux == j);
    Ndj = length(nObj_j);
    % For the left side of the equation
    auxLeft = Ndj*inv(params.invCj(:,:,j))+...
        (params.ap/params.bp)*Ndj*(params.mu_j(:,j)*params.mu_j(:,j)') +auxLeft;    
    % Rigth side of the equation
    sumXdn_j = sum(Xd(posObj,:),1)';
    auxRigth = auxRigth +sumXdn_j*params.mu_j(:,j)';
end

Wnew = (params.ap/params.bp)*auxRigth/(auxLeft);