function Nj_dn = Ndot_j_dn(S,d,n,j,params)
auxS = [];
for id = 1:params.D
    aux = S{id};
    if id == d
        aux(n) = [];
        auxS =[auxS;aux];
    else
        auxS =[auxS;aux];
    end
end

[valP,valI] = find(auxS == j);
Nj_dn = length(valP);


