# newtonMethodRev.jl
module simpleEKF

using LinearAlgebra

include("pricingFunctions.jl")
include("newtonMethod.jl")

using .newtonMethod
using .pricingFunctions

export kalman_filter_smoother_lag1

function kalman_filter_smoother_lag1(zAll, oIndAll, tcAll, I_z_t, f_t, n_c, n_p, n_s, n_t, n_u, n_x, n_z_t, AAll, BAll, DAll, GAll, Σw, Σv, a0, Σx, θF, θg, firstDates, tradeDates, ecbRatechangeDates, T0All,TAll)
    T = n_t;

    # Preallocate
    x_pred = [zeros(n_x) for _ in 1:T]
    P_pred = [zeros(n_x,n_x) for _ in 1:T]
    x_filt = [zeros(n_x) for _ in 1:T]
    P_filt = [zeros(n_x,n_x) for _ in 1:T]
    K      = [zeros(n_x,n_x) for _ in 1:T]

    oAll = [zeros(103, 22) for _ in 1:n_t]
    EAll = [zeros(3661, 6) for _ in 1:n_t]

    x_pred[1] = AAll[1]*Diagonal(θF)*BAll[1]*a0
    P_pred[1] = AAll[1]*Diagonal(θF)*BAll[1]*Σx*(AAll[1]*Diagonal(θF)*BAll[1])' + DAll[1]*Σw*DAll[1]'

    # Kalman Filter
    for t = 1:T
        oAll[t], EAll[t] = pricingFunctions.calcO(
            firstDates[t],
            tradeDates[t],
            θg,
            ecbRatechangeDates,
            n_c,
            n_z_t[t],
            T0All[t],
            TAll[t]
        )
        if t > 1
            x_pred[t] = AAll[t]*Diagonal(θF)*BAll[t] * x_filt[t-1]
            P_pred[t] = AAll[t]*Diagonal(θF)*BAll[t] * P_filt[t-1] * (AAll[t]*Diagonal(θF)*BAll[t])' + DAll[t]*Σw*DAll[t]'
        end
        H_t, u_t, g, Gradient = pricingFunctions.taylorApprox(oAll[t], oIndAll[t], tcAll[t], x_pred[t][1:n_s], I_z_t[t], n_z_t[t])
        R_t = GAll[t]*Σv*GAll[t]'
        K[t] = P_pred[t]*H_t' * inv(H_t*P_pred[t]*H_t' + R_t)

        innovation = vec(zAll[t]) - H_t*x_pred[t] - u_t
        x_filt[t] = x_pred[t] + K[t]*innovation
        P_filt[t] = (I - K[t]*H_t) * P_pred[t]
    end
    H_T, u_T, g, Gradient = pricingFunctions.taylorApprox(oAll[T], oIndAll[T], tcAll[T], x_pred[T][1:n_s], I_z_t[T], n_z_t[T])
    # RTS Smoother
    x_smooth = deepcopy(x_filt)
    P_smooth = deepcopy(P_filt)
    S = [zeros(n_x,n_x) for _ in 1:T-1]

    for t = T-1:-1:1
        S[t] = P_filt[t]*(AAll[t+1]*Diagonal(θF)*BAll[t+1])'*inv(P_pred[t+1])
        x_smooth[t] += S[t]*(x_smooth[t+1] - x_pred[t+1])
        P_smooth[t] += S[t]*(P_smooth[t+1] - P_pred[t+1])*S[t]'
    end

    # Lag-one covariance smoothing
    P_lag = [zeros(n_x,n_x) for _ in 1:T]
    P_lag[T] = (I - K[T]*H_T) * AAll[T]*Diagonal(θF)*BAll[T] * P_filt[T-1]

    for t = T-1:-1:2
        P_lag[t] = P_filt[t]*S[t-1]' + S[t]*(P_lag[t+1] - AAll[t+1]*Diagonal(θF)*BAll[t+1]*P_filt[t])*S[t-1]'
    end

    return x_filt, P_filt, x_smooth, P_smooth, P_lag, oAll, EAll
end

end # module
