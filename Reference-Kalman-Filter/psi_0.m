function [a_x, Sigma_x, Sigma_w, Sigma_v, theta_g, theta_F] = psi_0(wAll,vAll, xAll, thetaF, Q_K, x0)
    T = numel(wAll);

    max_W_len = max(cellfun(@(w) length(w), wAll));
    max_X_len = max(cellfun(@(x) length(x), xAll));
    max_V_len = max(cellfun(@(v) length(v), vAll));

    W=NaN(T, max_W_len);
    X=NaN(T, max_X_len);
    V=NaN(T, max_V_len);

    %wAll is a 5030x1 cell containing 50x1 double
    %xAll is a 5030x1 cell containing 50x1 double
    %vAll is a 5030x1 cell containing 28x1 double
    for t = 1:T
        W(t,1:length(wAll{t}))=wAll{t}(:)';
        X(t,1:length(xAll{t}))=xAll{t}(:)';
        V(t,1:length(vAll{t}))=vAll{t}(:)';
    end

    a_x = x0;
    Sigma_w = cov(W,'omitrows');
    Sigma_x = cov(X,'omitrows');
    Sigma_v = cov(V,'omitrows');
    theta_g = Q_K; 
    theta_F = thetaF;
end
