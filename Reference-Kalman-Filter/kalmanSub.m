function [KR, f0, fInit] = kalmanSub(K)

KR.a = cell(K.nt,1);
KR.A = cell(K.nt,1);
KR.b = cell(K.nt,1);
KR.B = cell(K.nt,1);
KR.c = cell(K.nt,1);
KR.e = cell(K.nt,1);
KR.E = cell(K.nt,1);
KR.h = cell(K.nt,1);
KR.H = cell(K.nt,1);

KR.deltav = cell(K.nt,1);
KR.deltaw = cell(K.nt,1);

KR.c_ = cell(K.nt,1);
KR.a_ = cell(K.nt,1);
KR.A_ = cell(K.nt,1);

KR.P_ = cell(K.nt,1);
KR.r_ = cell(K.nt,1);
KR.R_ = cell(K.nt,1);

KR.A_inva_ = cell(K.nt,1);
KR.A_invP_ = cell(K.nt,1);

KR.d = cell(K.nt,1);
KR.q = cell(K.nt,1);
KR.Q = cell(K.nt,1);

KR.dx = cell(K.nt,1);
KR.dw = cell(K.nt,1);
KR.dv = cell(K.nt,1);

c0 = 0; % Note that x0 term is missing
e0 = zeros(K.nx,1);
E0 = zeros(K.nx,K.nx);
fInit = c0;
cwAll = 0;
cvAll = 0;
cxAll = 0;
for t=K.nt:-1:1
  
  [KR.a{t}, KR.A{t}, KR.b{t}, KR.B{t}, KR.c{t}, KR.e{t}, KR.E{t}, KR.h{t}, KR.H{t}, cxAll, cwAll, cvAll] = K.approximate(t, cxAll, cwAll, cvAll);

  fInit = fInit + KR.c{t};

  if (t==1)
    KR.deltaw{t} = K.F{t}*K.x0 + K.w{t} - K.x{t};
  else
    KR.deltaw{t} = K.F{t}*K.x{t-1} + K.w{t} - K.x{t};
  end
  KR.deltav{t} = KR.h{t} + K.Iz{t}*K.x{t}(K.ns+1:end) + K.v{t} - K.z{t};

  Adeltaw = KR.A{t}*KR.deltaw{t};
  Bdeltav = KR.B{t}*KR.deltav{t};
  FA = K.F{t}'*KR.A{t};
  KR.c_{t} = KR.c{t} + (0.5*Adeltaw - KR.a{t})'*KR.deltaw{t} + (0.5*Bdeltav- KR.b{t})'*KR.deltav{t};
  KR.a_{t} = KR.a{t} - Adeltaw + KR.H{t}'*(Bdeltav - KR.b{t}) + KR.e{t};
  KR.A_{t} = KR.A{t} + KR.H{t}'*KR.B{t}*KR.H{t} + KR.E{t};
  if (t < K.nt)
    KR.c_{t} = KR.c_{t} + KR.d{t+1};
    KR.a_{t} = KR.a_{t} + KR.q{t+1};
    KR.A_{t} = KR.A_{t} + KR.Q{t+1};   
  end
  KR.P_{t} = FA;
  KR.r_{t} = K.F{t}'*(Adeltaw - KR.a{t});
  KR.R_{t} = FA*K.F{t};
  
%   lambda = eigs(KR.A_{t},5,'smallestabs');
%   if (lambda < 0)
%     KR.A_{t} = KR.A_{t} - eye(size(KR.A_{t}))*lambda*1.1;
%   end
  
  KR.A_inva_{t} = KR.A_{t}\KR.a_{t};
  KR.A_invP_{t} = KR.A_{t}\(KR.P_{t}');
  
  KR.d{t} = KR.c_{t} - 0.5* KR.a_{t}'*KR.A_inva_{t};
  KR.q{t} = KR.r_{t} + KR.P_{t}*KR.A_inva_{t};
  KR.Q{t} = KR.R_{t} - KR.P_{t}*KR.A_invP_{t};
end

KR.c0_ = c0 + KR.d{1};
KR.a0_ = e0 + KR.q{1};
KR.A0_ = E0 + KR.Q{1};

KR.dx0 = -KR.A0_\KR.a0_;
f0 = KR.c0_ + 0.5*KR.a0_'*KR.dx0;

fprintf('%d %d %d %d %d %d\n', cwAll, cvAll, cxAll, fInit, f0, f0-fInit);

for t=1:K.nt
  if (t==1)
    KR.dx{t} = KR.A_invP_{t} * KR.dx0 - KR.A_inva_{t};
    KR.dw{t} = KR.dx{t} - K.F{t} * KR.dx0 - KR.deltaw{t};    
  else
    KR.dx{t} = KR.A_invP_{t} * KR.dx{t-1} - KR.A_inva_{t};      
    KR.dw{t} = KR.dx{t} - K.F{t} * KR.dx{t-1} - KR.deltaw{t};
  end
  KR.dv{t} = - KR.H{t} * KR.dx{t} - KR.deltav{t};
end

f1 = zeros(K.nt,1);
f2 = zeros(K.nt,1);
f3 = zeros(K.nt,1);
for t=1:K.nt
  f1(t) = KR.c{t} + KR.a{t}'*KR.dw{t} + 0.5*KR.dw{t}'*KR.A{t}*KR.dw{t} + KR.b{t}'*KR.dv{t} + 0.5*KR.dv{t}'*KR.B{t}*KR.dv{t} + KR.e{t}'*KR.dx{t} + 0.5*KR.dx{t}'*KR.E{t}*KR.dx{t};
  if (t<K.nt)
    f1(t) = f1(t) + KR.d{t+1} + KR.q{t+1}'*KR.dx{t} + 0.5*KR.dx{t}'*KR.Q{t+1}*KR.dx{t};
  end
  f2(t) = KR.c_{t} + KR.a_{t}'*KR.dx{t} + 0.5*KR.dx{t}'*KR.A_{t}*KR.dx{t};
  f3(t) = KR.d{t};
  if (t==1)
    f2(t) = f2(t) + KR.r_{t}'*KR.dx0 + 0.5*KR.dx0'*KR.R_{t}*KR.dx0 - KR.dx0'*KR.P_{t}*KR.dx{t};
    f3(t) = f3(t) + KR.q{t}'*KR.dx0 + 0.5*KR.dx0'*KR.Q{t}*KR.dx0;
  else
    f2(t) = f2(t) + KR.r_{t}'*KR.dx{t-1} + 0.5*KR.dx{t-1}'*KR.R_{t}*KR.dx{t-1} - KR.dx{t-1}'*KR.P_{t}*KR.dx{t};
    f3(t) = f3(t) + KR.q{t}'*KR.dx{t-1} + 0.5*KR.dx{t-1}'*KR.Q{t}*KR.dx{t-1};
  end
end

% plot(f1-f2);
