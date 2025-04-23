% Class for Kalman filter of interest rates

classdef clsKalmanIr < handle
   properties
      nt               = 0   %
      np               = 0   %
      nc               = 0   %
      ns               = 0   %
      nu               = 0   %
      nx               = 0   %
      nz               = 0   %

      x0               = 0   %
      x                = 0   %
      w                = 0   %
      z                = 0   %
      v                = 0   %
      F                = 0   %
      Iz               = 0   %
      Cw               = 0   %
      Cx               = 0   %
      Cv               = 0   %
      nuv              = 0   %

      thetaF           = 0   %
      Sigmaw           = 0   %
      Sigmav           = 0   %
      ax               = 0   % E[x0]
      Sigma0           = 0   % Cov[x0]
      Sigmax           = 0   % Cov[xt]

      % Matrices for keeping track of parameters
      A                = 0  % ThetaF - Rows
      B                = 0  % ThetaF - Columns
      D                = 0  % Sigmaw
      G                = 0  % Sigmav

      vStudentsT       = 0   %
      pl               = 0   %
      atAll            = 0   %
      oAll             = 0   %
      oIndAll          = 0   %
      tcAll            = 0   %
   end
   
   methods
      
      function K = clsKalmanIr(times, Ef, ef, fH, indInstrAll, usedInstr, priceAll, kx, pl, atAll, oAll, oIndAll, tcAll, nSteps, cbEffectiveDates, thetaF, Sigmaw, Sigmav, Sigmax, ax, Sigma0)
        K.vStudentsT = true;

        K.nt = length(times);
        K.np = size(Ef, 2);
        K.nc = nSteps;
        K.ns = K.np+K.nc;
        K.nu = sum(usedInstr);
        K.nx = K.np + K.nc + K.nu;
        K.nz = zeros(K.nt, 1);
        
        K.pl = pl;
        K.atAll = atAll;
        K.oAll = oAll;
        K.oIndAll = oIndAll;
        K.tcAll = tcAll;       

        K.x0 = zeros(K.nx,1);
        K.x = cell(K.nt,1);
        K.w = cell(K.nt,1);
        K.z = cell(K.nt,1);
        K.v = cell(K.nt,1);
        K.F = cell(K.nt,1);
        K.Iz = cell(K.nt,1);
        K.Cw = cell(K.nt,1);
        K.Cx = cell(K.nt,1);
        K.Cv = cell(K.nt,1);
        K.nuv = cell(K.nt,1);

        % Parameters for Kalman filter

        K.thetaF = thetaF;
        K.Sigmaw = Sigmaw;
        K.Sigmav = Sigmav;
        K.Sigmax = Sigmax;
        K.ax = ax;
        K.Sigma0 = Sigma0;
        
        
        K.thetaF = ones(K.nx,1);
        K.Sigmaw = [ef*kx(1) ; ones(K.nc,1) ; ones(K.nu,1)*kx(2)];
        K.Sigmav = ones(K.nu,1)*kx(4);
        K.ax = zeros(K.nx,1);      % E[x0]
        K.Sigma0 = eye(K.nx,1);   % Cov[x0] - denoted Sigmax by Carl & Axel - should be changed to Sigma0
        K.Sigmax = [zeros(K.np,1) ; zeros(K.nc,1) ; ones(K.nu,1)*kx(3)];
        
        % Matrices for keeping track of parameters
        K.A = cell(K.nt,1); % ThetaF - Rows
        K.B = cell(K.nt,1); % ThetaF - Columns
        K.D = cell(K.nt,1); % Sigmaw
        K.G = cell(K.nt,1); % Sigmav
        
        
        te = find(cbEffectiveDates > floor(times(1)));
        te = te(1);
        
        xpInit = fH(:,1:size(Ef,1))*Ef;
        for t=1:K.nt
          K.nz(t) = length(indInstrAll{t});
          K.x{t} = zeros(K.nx,1);
          K.w{t} = zeros(K.nx,1);
          K.z{t} = priceAll{t};
          K.v{t} = zeros(K.nz(t),1);
          if (floor(times(t)) >= cbEffectiveDates(te))
%             F = zeros(K.nx);
%             F(1:K.np, 1:K.np) = eye(K.np);
%             for i=1:K.nc-1
%               F(K.np+i, K.np+i+1) = 1;
%             end
%             F(K.ns+(1:K.nu), K.ns+(1:K.nu)) = eye(K.nu);
%             K.F{t} = sparse(F);
            te = te+1;
            K.A{t} = sparse(eye(K.nx));
            B = eye(K.nx);
            for i=(K.np+1):(K.np+K.nc)
              if (i < K.np+K.nc)
                B(i,i+1) = B(i,i);
              end
              B(i,i) = 0;
            end
            K.B{t} = sparse(B);
            
%             K.F{t} = sparse(eye(K.nx));
          else
%             K.F{t} = sparse(eye(K.nx));
            K.A{t} = sparse(eye(K.nx));
            K.B{t} = sparse(eye(K.nx));
          end
          K.Iz{t} = sparse(1:K.nz(t), indInstrAll{t}, ones(K.nz(t),1), K.nz(t), K.nu);

%           K.Cw{t} = [ef*kx(1) ; ones(K.nc,1) ; ones(K.nu,1)*kx(2)];
%           K.Cx{t} = [zeros(K.np,1) ; zeros(K.nc,1) ; ones(K.nu,1)*kx(3)];
          K.Cx{t} = K.Sigmax;
%           K.Cv{t} = ones(K.nz(t),1)*kx(4);
          if (K.vStudentsT)
            K.nuv{t} = ones(K.nz(t),1)*8;
          end
          K.x{t}(1:K.np) = xpInit(t,:)';
          
          F = K.A{t}*diag(K.thetaF)*K.B{t};
          K.F{t} = sparse(F);
%           if (norm(F-K.F{t}) > 1E-5)
%             error('Differ');
%           end
          
          K.G{t} = sparse(1:K.nz(t), indInstrAll{t}, ones(K.nz(t),1), K.nz(t), K.nu);
          Cv = K.G{t}*diag(K.Sigmav)*K.G{t}';
%           if (norm(diag(Cv)-K.Cv{t}) > 1E-5)
%             error('Differ');
%           end
          K.Cv{t} = diag(Cv);
          
          K.D{t} = sparse(eye(K.nx));
          Cw = K.D{t}*diag(K.Sigmaw)*K.D{t}';
%           if (norm(diag(Cw)-K.Cw{t}) > 1E-5)
%             error('Differ');
%           end
          K.Cw{t} = diag(Cw);
          
        end

      end % clsPriceStochastic constructor
      
      function [a, A, b, B, c, e, E, h, H, cxAll, cwAll, cvAll] = approximate(K, t, cxAll, cwAll, cvAll)
        % Second order Taylor approximation of normal pdf
        ind = (K.Cx{t}~=0);
      %   c = -sum(log(normpdf(K.w{t}, 0, sqrt(K.Cw{t}))))-sum(log(normpdf(K.x{t}(ind), 0, sqrt(K.Cx{t}(ind)))));
        c = -sum(-0.5*log(2*pi*K.Cw{t})-0.5*(K.w{t}./sqrt(K.Cw{t})).^2)-sum(-0.5*log(2*pi*K.Cx{t}(ind))-0.5*(K.x{t}(ind)./sqrt(K.Cx{t}(ind))).^2);
      cwAll = cwAll - sum(-0.5*log(2*pi*K.Cw{t})-0.5*(K.w{t}./sqrt(K.Cw{t})).^2);
      cxAll = cxAll - sum(-0.5*log(2*pi*K.Cx{t}(ind))-0.5*(K.x{t}(ind)./sqrt(K.Cx{t}(ind))).^2);

        if (K.vStudentsT) % Student's t-distribution
          vpdf = pdf('tLocationScale',K.v{t}, 0, sqrt(K.Cv{t}), K.nuv{t});
      %     vpdf2 = gamma((K.nuv{t}+1)/2)./(gamma(K.nuv{t}/2).*sqrt(K.nuv{t}*pi).*sqrt(K.Cv{t})).*(1 + K.v{t}.^2./(K.nuv{t}.*K.Cv{t})).^(-(K.nuv{t}+1)/2);
          c = c - sum(log(vpdf));
      cvAll = cvAll - sum(log(vpdf));
          k = gamma((K.nuv{t}+1)/2)./(gamma(K.nuv{t}/2).*sqrt(K.nuv{t}*pi).*sqrt(K.Cv{t}));
          dvpdf = -k.*(1 + K.v{t}.^2./(K.nuv{t}.*K.Cv{t})).^(-(K.nuv{t}+3)/2) .* (K.nuv{t}+1) .* K.v{t}./(K.nuv{t}.*K.Cv{t});
          d2vpdf = k.*(1 + K.v{t}.^2./(K.nuv{t}.*K.Cv{t})).^(-(K.nuv{t}+5)/2) .* (K.nuv{t}+3) .* (K.nuv{t}+1) .* K.v{t}.^2./(K.nuv{t}.^2.*K.Cv{t}.^2) ...
                    -k.*(1 + K.v{t}.^2./(K.nuv{t}.*K.Cv{t})).^(-(K.nuv{t}+3)/2) .* (K.nuv{t}+1)./(K.nuv{t}.*K.Cv{t});
          b = -dvpdf./vpdf;
          B = diag(dvpdf.^2./vpdf.^2 - d2vpdf./vpdf);

        else % Normal distribution
      %     c = c - sum(log(normpdf(K.v{t}, 0, sqrt(K.Cv{t}))));
          c = c - sum(-0.5*log(2*pi*K.Cv{t})-0.5*(K.v{t}./sqrt(K.Cv{t})).^2);
      cvAll = cvAll - sum(-0.5*log(2*pi*K.Cv{t})-0.5*(K.v{t}./sqrt(K.Cv{t})).^2);    
          b = K.v{t}./K.Cv{t};
          B = diag(1./K.Cv{t});
        end
        
        a = K.w{t}./K.Cw{t};
        A = diag(1./K.Cw{t});
        tmp = zeros(K.nx,1);
        tmp(ind) = K.x{t}(ind)./K.Cx{t}(ind);
        e = tmp;
        tmp(ind) = 1./K.Cx{t}(ind);
        E = diag(tmp);

        if (isinf(c))
          hej = 1
        end
        
        [h, G] = taylorInstrPC(K.pl, K.atAll{t}, K.oAll{t}, K.oIndAll{t}, K.tcAll{t}, K.x{t}(1:K.ns));
        H = [G K.Iz{t}];
      end

      function [f] = evalObj(K)
        % Evaluate objective function value
        f = 0; % Note that x0 term is missing
        
        for t=1:K.nt
          ind = (K.Cx{t}~=0);
%           f = f - sum(log(normpdf(K.w{t}, 0, sqrt(K.Cw{t}))))-sum(log(normpdf(K.x{t}(ind), 0, sqrt(K.Cx{t}(ind)))));
          f = f - sum(-0.5*log(2*pi*K.Cw{t})-0.5*(K.w{t}./sqrt(K.Cw{t})).^2)-sum(-0.5*log(2*pi*K.Cx{t}(ind))-0.5*(K.x{t}(ind)./sqrt(K.Cx{t}(ind))).^2);
          if (K.vStudentsT) % Student's t-distribution
            vpdf = pdf('tLocationScale',K.v{t}, 0, sqrt(K.Cv{t}), K.nuv{t});
            f = f - sum(log(vpdf));

          else % Normal distribution
%             f = f - sum(log(normpdf(K.v{t}, 0, sqrt(K.Cv{t}))));
            f = f - sum(-0.5*log(2*pi*K.Cv{t})-0.5*(K.v{t}./sqrt(K.Cv{t})).^2);
          end
        end
      end
      
      
      function [D] = dividends(obj, dm, dc)
        Nc = length(dm.cName);
        D = cell(Nc, 1);
      end

      function [iCurPrice] = priceCurrency(obj)
        iCurPrice = obj.iCurPrice;
      end
   end
   
   methods (Access = 'private') % Access by class members only
   end
end % classdef





