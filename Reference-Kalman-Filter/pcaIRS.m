function [fH, piH, lastDates, nIn, ef, Ef, fTotVar, ePi, EPi, piTotVar] = pcaIRS(times, fH, piH, firstDates, lastDates, outOfSampleStartDate)

nD = max(lastDates-firstDates);

% Extrapolate interest rate curves (deals with issue that length varies due to holidays and weekends) - required for performance attribution
nStartExtrapolate = min(lastDates-firstDates)+1;
for i = nStartExtrapolate:nD
  ind = (lastDates-firstDates < i);
  fH(ind, i) = fH(ind, i-1);
  piH(ind, i) = piH(ind, i-1);
end
lastDates = firstDates + nD;

nIn = length(find(times<outOfSampleStartDate)); % In-sample
fprintf('In-sample range: %s to %s\n', datestr(floor(times(1))), datestr(floor(times(nIn))))
fprintf('Out-of-sample range: %s to %s\n', datestr(floor(times(nIn+1))), datestr(floor(times(end))))

nH = min(lastDates-firstDates);
% r = fH(2:end,1:nH)-fH(1:end-1,1:nH);
r = fH(2:nIn,1:nH)-fH(1:nIn-1,1:nH);
r = r - repmat(mean(r, 1), size(r,1), 1);
C = cov(r);
nEigs = 6;

[V,D] = eigs(C, nEigs);
[ef,ind] = sort(diag(D),1, 'descend');
Ef = V(:,ind);

ET = (1:size(Ef,1))'/365;
cef = cumsum(ef);
fTotVar = sum(diag(C));
h = figure(1);
plot(ET, Ef(:,1), 'b', ET, Ef(:,2), 'g', ET, Ef(:,3), 'r', ET, Ef(:,4), 'b--', ET, Ef(:,5), 'g--', ET, Ef(:,6), 'r--');
title('PCA forward rates')
shift     = sprintf('Shift        %5.2f%% (%5.2f%%)\n',100*ef(1)/fTotVar, 100*cef(1)/fTotVar);
twist     = sprintf('Twist        %5.2f%% (%5.2f%%)\n',100*ef(2)/fTotVar, 100*cef(2)/fTotVar);
butterfly = sprintf('Butterfly    %5.2f%% (%5.2f%%)\n',100*ef(3)/fTotVar, 100*cef(3)/fTotVar);
PC4       = sprintf('Loadings PC4 %5.2f%% (%5.2f%%)\n',100*ef(4)/fTotVar, 100*cef(4)/fTotVar);
PC5       = sprintf('Loadings PC5 %5.2f%% (%5.2f%%)\n',100*ef(5)/fTotVar, 100*cef(5)/fTotVar);
PC6       = sprintf('Loadings PC6 %5.2f%% (%5.2f%%)\n',100*ef(6)/fTotVar, 100*cef(6)/fTotVar);
legend(shift,twist,butterfly, PC4, PC5, PC6, 'Location', 'Best');
% set(h,'Units','Inches');
% pos = get(h,'Position');
% set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3),pos(4)]);
% saveas(gcf,'fig/ForwPcaEua.pdf')

% [E, lambda] = mexPortfolio('nLargestEigenvalues', r, nEigs);
% 
% figure(2)
% plot(E)

% r = piH(2:end,1:nH)-piH(1:end-1,1:nH);
r = piH(2:nIn,1:nH)-piH(1:nIn-1,1:nH);
r = r - repmat(mean(r, 1), size(r,1), 1);
C = cov(r);
[V,D] = eigs(C, nEigs);
[ePi,ind] = sort(diag(D),1, 'descend');
EPi = V(:,ind);

cePi = cumsum(ePi);
piTotVar = sum(diag(C));
h = figure(2);
plot(ET, EPi(:,1), 'b', ET, EPi(:,2), 'g', ET, EPi(:,3), 'r', ET, EPi(:,4), 'b--', ET, EPi(:,5), 'g--', ET, EPi(:,6), 'r--');
title('PCA spread \pi')
% shift     = sprintf('Shift        %5.2f%% (%5.2f%%)\n',100*ePi(1)/piTotVar, 100*cePi(1)/piTotVar);
% twist     = sprintf('Twist        %5.2f%% (%5.2f%%)\n',100*ePi(2)/piTotVar, 100*cePi(2)/piTotVar);
% butterfly = sprintf('Butterfly    %5.2f%% (%5.2f%%)\n',100*ePi(3)/piTotVar, 100*cePi(3)/piTotVar);
shift     = sprintf('Loadings PC1 %5.2f%% (%5.2f%%)\n',100*ePi(1)/piTotVar, 100*cePi(1)/piTotVar);
twist     = sprintf('Loadings PC2 %5.2f%% (%5.2f%%)\n',100*ePi(2)/piTotVar, 100*cePi(2)/piTotVar);
butterfly = sprintf('Loadings PC3 %5.2f%% (%5.2f%%)\n',100*ePi(3)/piTotVar, 100*cePi(3)/piTotVar);
PC4       = sprintf('Loadings PC4 %5.2f%% (%5.2f%%)\n',100*ePi(4)/piTotVar, 100*cePi(4)/piTotVar);
PC5       = sprintf('Loadings PC5 %5.2f%% (%5.2f%%)\n',100*ePi(5)/piTotVar, 100*cePi(5)/piTotVar);
PC6       = sprintf('Loadings PC6 %5.2f%% (%5.2f%%)\n',100*ePi(6)/piTotVar, 100*cePi(6)/piTotVar);
legend(shift,twist,butterfly, PC4, PC5, PC6, 'Location', 'Best');
% set(h,'Units','Inches');
% pos = get(h,'Position');
% set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3),pos(4)]);
% saveas(gcf,'fig/piPcaEua.pdf')




% xi = [fH(:,1:nH)*Ef piH(:,1:nH) * EPi];
% for i=1:250
%   plot([fH(i+1,1:nH) - fH(i,1:nH) ; (xi(i+1,1:6)-xi(i,1:6))*Ef']')
%   pause;
% end

