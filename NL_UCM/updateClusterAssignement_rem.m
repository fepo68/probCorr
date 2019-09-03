function S = updateClusterAssignement_rem(S,params,iRem)

for i=1:params.D
    Sd = S{i};
    [pos,ip] = find(Sd>iRem);
    Sd(pos) = Sd(pos) - 1;
    S{i} = Sd;
    
end