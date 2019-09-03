function Nj = Ndot_j(S,j,params)
auxS = [];
for id = 1:params.D
    aux = S{id};
    
    auxS =[auxS;aux];
    
end

[valP,valI] = find(auxS == j);
Nj = length(valP);


