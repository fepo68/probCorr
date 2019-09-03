function Nj = Ndjval(S,j,d)
auxS = [];
id = d;
aux = S{id};

auxS =[auxS;aux];

% end

[valP,valI] = find(auxS == j);
Nj = length(valP);


