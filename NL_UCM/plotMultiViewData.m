function plotMultiViewData(Y,Sclust)
cmap = colormap('lines');
V = length(Y);
% figure
shapeV = {'o','s','d','*','+'};
for v = 1:V
    Yd = Y{v};
    Sd = Sclust{v};
    K = length(unique(Sd));
    [Nd,Dd] = size(Yd);
    subplot(1,V,v),
    legendInfo = {};
    for c =1:K
        [~,pos] = find(Sd==c);
        plot(Yd(pos,1),Yd(pos,2), shapeV{v},'Color',cmap(c,:),'MarkerFaceColor',cmap(c,:));
        hold on
        legendInfo{c} = ['Cluster' num2str(c)]; % or whatever is appropriate
    end
    legend(legendInfo);
    title(['Data (view ',num2str(v),')']);
end
