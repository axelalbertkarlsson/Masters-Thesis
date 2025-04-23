% File that was missing before
function [g, G] = taylorInstrPC(pl, at, o, oInd, tc, x)
nInstr = length(at);
g = zeros(nInstr,1);
G = zeros(nInstr,length(x));

for j=1:nInstr
  ind = oInd(j):(oInd(j+1)-1);
  if (at(j) == pl.atIBOR)
    eox = exp(-o(ind,:)*x);
    g(j) = (eox-1)/tc(ind);
    G(j,:) = -eox*o(ind,:)/tc(ind);
  elseif (at(j) == pl.atOISG)
    eox = exp(o(ind,:)*x);
    den = tc(ind(2:end))'*eox(2:end);
    g(j) = (eox(1)-eox(end))/den;
    G(j,:) = (eox(1)*o(ind(1),:)-eox(end)*o(ind(end),:))/den - (eox(1)-eox(end))/den^2*((tc(ind(2:end)).*eox(2:end))'*o(ind(2:end),:));
  end
end
