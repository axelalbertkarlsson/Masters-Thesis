% Ricatti recursion

kx = [1E5 ; 1E0 ; 1E2 ; 1E1];

np = size(Ef, 2);
nc = nSteps;
nu = sum(usedInstr);
nx = np + nc + nu;

thetaF = ones(nx,1);
Sigmaw = [ef*kx(1) ; ones(nc,1) ; ones(nu,1)*kx(2)];
Sigmav = ones(nu,1)*kx(4);
ax = zeros(nx,1);      % E[x0]
Sigma0 = eye(nx,1);   % Cov[x0] - denoted Sigmax by Carl & Axel - should be changed to Sigma0
Sigmax = [zeros(np,1) ; zeros(nc,1) ; ones(nu,1)*kx(3)];

% runReducedIrCurvesInit;

K = clsKalmanIr(times, Ef, ef, fH, indInstrAll, usedInstr, priceAll, kx, pl, atAll, oAll, oIndAll, tcAll, nSteps, cbEffectiveDates, thetaF, Sigmaw, Sigmav, Sigmax, ax, Sigma0); 

for iter=1:5

  [KR, f0, fInit] = kalmanSub(K);
  
  if (iter <= 100)
    lambda = 1;
  elseif (iter < 10)
    lambda = 0.01;  
  else
    lambda = 1;  
  end
  K.x0 = K.x0 + lambda*KR.dx0;
  for t=1:K.nt
    K.x{t} = K.x{t} + lambda*KR.dx{t};
    K.w{t} = K.w{t} + lambda*KR.dw{t};
    K.v{t} = K.v{t} + lambda*KR.dv{t};

%     if (t>1)
%       diff(t) = norm(K.F{t}*K.x{t-1} + K.w{t} - K.x{t});
%     end
  end
  fNew = evalObj(K);
  fprintf('%.16d %.16d %d\n', f0, fNew, fNew - f0);
  if (abs(f0-fInit)< 1E-8)
    break;
  end

end

xp = zeros(K.nt, K.np);
xc = zeros(K.nt, K.nc);
xu = zeros(K.nt, K.nu);
v = zeros(K.nt, K.nu);
w = zeros(K.nt, K.nx);
for t=1:K.nt
  xp(t,:) = K.x{t}(1:K.np);
  xc(t,:) = K.x{t}(K.np + (1:K.nc));
  xu(t,:) = K.x{t}(K.ns+1:end);
  v(t,indInstrAll{t}) = K.v{t};
  w(t,:) = K.w{t};
end

%% Figure

figure(1);
plot(xp(:,1:3));
title('x PC');

figure(2);
plot(xc(:,1:3));
title('x step');

figure(3);
plot(xu);
title('x unsystematic');

figure(4);
plot(v);
title('v');

figure(5);

% for k=1:K.nt
% % for k=4300:K.nt
% 
%   tradeDate = floor(times(k));
%   datesStep = cbEffectiveDates(cbEffectiveDates > tradeDate);
%   datesStep = datesStep(1:nSteps);
%   
%   Es = zeros(size(Ef,1), nSteps);
%   for i=1:nSteps
%     Es(datesStep(i)-tradeDate+1:end, i) = 1;
%   end
%   E = [Ef Es];
%    
%   plot((0:size(E,1)-1)/365, E*[xp(k,:) xc(k,:)]', (0:size(fH,2)-1)/365, fH(k,:));
%   title(datestr(floor(times(k))));
%   pause(0.01);
% 
%   if (false)
%   figure(5+mod(k,2))
%   plot((0:size(E,1)-1)/365, E(:,7:end)*[xc(k,:)]');
%   title(datestr(floor(times(k))));
%   hold off;
% 
%   figure(5+mod(k+1,2))
%   plot((0:size(E,1)-1)/365, E(:,7:end)*[xc(k,:)]');
% %   title(datestr(floor(times(k))));
%   hold on;
%   
%   pause;
%   end
% end
