function pliN = fixPli(pli)

[N,K] = size(pli);
pliN = zeros(N,K);
indL = 1:K;
for n = 1:N
    pl = pli(n,:);
    [~,bb] = max(pl);
    pl(bb) = .8;
    indit = indL;
    indit(bb) = [];
    pl(indit) = (1-.8)/(K-1);
    pliN(n,:) = pl;
end