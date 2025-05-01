using Revise, LinearAlgebra, Plots

include("loadData.jl")
include("pricingFunctions.jl")
include("newtonMethod.jl")
include("outputData.jl")
include("plots.jl")

using .plots
using .outputData
using .newtonMethod
using .loadData
using .pricingFunctions

# Load data
data = loadData.run()

function kalman_filter_smoother_lag1(zAll, oAll, oIndAll, tcAll, I_z_t, f_t, n_c, n_p, n_s, n_t, n_u, n_x, n_z_t, AAll, BAll, DAll, GAll, Σw, Σv, a0, Σx, θF, θg)
    T = n_t;

    # Preallocate
    x_pred = [zeros(n_x) for _ in 1:T]
    P_pred = [zeros(n_x,n_x) for _ in 1:T]
    x_filt = [zeros(n_x) for _ in 1:T]
    P_filt = [zeros(n_x,n_x) for _ in 1:T]
    K      = [zeros(n_x,n_x) for _ in 1:T]

    x_pred[1] = AAll[1]*Diagonal(θF)*BAll[1]*a0
    P_pred[1] = AAll[1]*Diagonal(θF)*BAll[1]*Σx*(AAll[1]*Diagonal(θF)*BAll[1])' + DAll[1]*Σw*DAll[1]'

    # Kalman Filter
    for t = 1:T
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

    return x_filt, P_filt, x_smooth, P_smooth, P_lag
end


# Initialize 
oAll = [zeros(103, 22) for _ in 1:data.n_t]
EAll = [zeros(3661, 6) for _ in 1:data.n_t]

# Calculate oAll
for t in 1:Int(data.n_t)
    oAll[t], EAll[t] = pricingFunctions.calcO(
        data.firstDates[t],
        data.tradeDates[t],
        data.theta_g,
        data.ecbRatechangeDates,
        data.n_c,
        data.n_z_t[t],
        data.T0All[t],
        data.TAll[t]
    )
end

x_filt, P_filt, x_smooth, P_smooth, P_lag = kalman_filter_smoother_lag1(data.zAll, oAll, data.oIndAll, data.tcAll, data.I_z_t, data.f_t, Int(data.n_c), Int(data.n_p), Int(data.n_s), Int(data.n_t), Int(data.n_u), Int(data.n_x), Int.(data.n_z_t), data.A_t, data.B_t, data.D_t, data.G_t, data.Sigma_w, data.Sigma_v, vec(data.a_x), data.Sigma_x, vec(data.theta_F), data.theta_g);


fAll, priceAll = outputData.calculateRateAndRepricing(
    EAll, data.zAll, data.I_z_t, x_smooth, oAll, data.oIndAll, data.tcAll, data.theta_g, Int.(data.n_z_t), Int(data.n_t), Int(data.n_s)
)

T_new = Int(data.n_t/2)

plots.plot3DCurve(data.times, fAll)




# function dummy_nll(ψ)
#     x, y = ψ
#     return x^2 + 2y^2
# end

# ψ₀ = [4.0, -5.0]
# ψ_opt = newtonMethod.newtonOptimize(dummy_nll, ψ₀; verbose=true)

# println("\nOptimized ψ = ", ψ_opt)

