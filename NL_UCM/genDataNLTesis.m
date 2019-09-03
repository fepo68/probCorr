%%

kern = kernCreate(Wd*Z(find(s_dn),:)','rbf');

k = kernCompute(kern,Wd*Z(find(s_dn),:)');

mu = Wd*Z(find(s_dn),:)';


xNL = gsamp(mu,k,1)