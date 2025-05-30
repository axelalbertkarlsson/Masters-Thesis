module outputData


include("pricingFunctions.jl")

using LinearAlgebra, Printf, MAT
using .pricingFunctions

import Statistics

export calculateMSE, calculateRateAndRepricing, write_results, kalman_filter_xpinit

function calculateRateAndRepricing(EAll, zAll, I_z_t, xAll, oAll, oIndAll, tcAll, θg, n_z_t,n_t, n_s, n_u,  GAll, Σv, P_predAll)
    # Initialization
    T=n_t;
    E = EAll[1]
    n_rows = size(θg, 1)  # 3661, consistent with the rest
    fAll = zeros(n_rows, T)    
    zPredAll = Vector{Vector{Float64}}(undef, T)
    innovationAll = deepcopy(zAll);
    innovationLik = zeros(size(zAll,1),1);
    last_lik = 0.0

    for t in 1:T
        x = xAll[t]
        x_s = x[1:n_s]

        H_t, u_t, g, Gradient = pricingFunctions.taylorApprox(oAll[t], oIndAll[t], tcAll[t], x_s, I_z_t[t], n_z_t[t])

        E = EAll[t];

        E = E[end - 3660:end, :]  # Keep only the last 3661 rows, otherwise mismatch

        # println(size(E))
        # println(size(θg))
        # println(size(x[1:(n_s+6)]))

        fAll[:, t] = [θg E] * x[1:(n_u)]

        zPredAll[t] = H_t*x + u_t

        innovationAll[t] = zAll[t] - H_t*x - u_t #Maybe zAll[t] - H_t*x_pred -u_t

        # 2) build the innovation covariance
        S = H_t * P_predAll[t] * H_t' +
        GAll[t] * Σv * GAll[t]'

        try
            ld = logdet(S)                           # may still throw
            quad = innovationAll[t]' * (S \ innovationAll[t])
            innovationLik[t] = -0.5 * (n_z_t[t]*log(2π) + ld + quad)
            last_lik = innovationLik[t]             # update fallback
        catch e
            @warn "Failure at t=$t computing likelihood: $e. Using fallback = $last_lik"
            # fall back to the mean of all previously‐computed liks (or last_lik)
            innovationLik[t] = t > 1 ? Statistics.mean(innovationLik[1:t-1]) : last_lik
        end

    end

    return fAll, zPredAll, innovationAll, innovationLik;
end

function safe_inv_spd!(S::AbstractMatrix{T}) where T<:Real
    # add jitter until S is finite and SPD
    jitter = 1e-8
    for _ in 1:5
      if all(isfinite, S)
        # try a cheap SPD check
        try 
          cholesky(S; check=true)
          return S
        catch
        end
      end
      S .+= jitter .* I(size(S,1))
      jitter *= 10
    end
  end

function kalman_filter_xpinit(zAll, oIndAll, tcAll, I_z_t, f_t, n_c, n_p, n_s, n_t, n_u, n_x, n_z_t, AAll, BAll, DAll, GAll, Σw, Σv, a0, Σx, θF, θg, firstDates, tradeDates, ecbRatechangeDates, T0All,TAll)
    T = n_t;

    # Preallocate
    x_pred = [zeros(n_x) for _ in 1:T]
    P_pred = [zeros(n_x,n_x) for _ in 1:T]
    x_filt = [zeros(n_x) for _ in 1:T]
    P_filt = [zeros(n_x,n_x) for _ in 1:T]
    K      = [zeros(n_x,n_x) for _ in 1:T]

    oAll = [zeros(103, 22) for _ in 1:T]
    EAll = [zeros(3661, 6) for _ in 1:T]

    x_pred[1] = AAll[1]*Diagonal(θF)*BAll[1]*a0
    P_pred[1] = AAll[1]*Diagonal(θF)*BAll[1]*Σx*(AAll[1]*Diagonal(θF)*BAll[1])' + DAll[1]*Σw*DAll[1]'
    xpInit = f_t[:,1:size(θg,1)]*θg
    x_pred[1][1:n_p] = xpInit[1,:]'
    # Kalman Filter
    for t = 1:T
        if t > 1
            x_pred[t] = AAll[t]*Diagonal(θF)*BAll[t] * x_filt[t-1]
            P_pred[t] = AAll[t]*Diagonal(θF)*BAll[t] * P_filt[t-1] * (AAll[t]*Diagonal(θF)*BAll[t])' + DAll[t]*Σw*DAll[t]'
        end
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
        # New addition
        x_pred[t][1:n_p] = xpInit[t,:]'
        # New addition ^^
        H_t, u_t, g, Gradient = pricingFunctions.taylorApprox(oAll[t], oIndAll[t], tcAll[t], x_pred[t][1:n_s], I_z_t[t], n_z_t[t])
        R_t = GAll[t]*Σv*GAll[t]'
        S_t = H_t*P_pred[t]*H_t' + R_t
        safe_inv_spd!(S_t)
        # now invert safely
        K[t] = P_pred[t]*H_t' * inv(S_t)
        #K[t] = P_pred[t]*H_t' * inv(H_t*P_pred[t]*H_t' + R_t)

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
        #S[t] = P_filt[t]*(AAll[t+1]*Diagonal(θF)*BAll[t+1])'*inv(P_pred[t+1])
        S[t] = P_filt[t]*(AAll[t+1]*Diagonal(θF)*BAll[t+1])' * inv(P_pred[t+1])

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

function calculateMSE(innovationAll)
    # === Print overall error metrics for both methods ===
    # 1) flatten into one long numeric vector
    all_reg = vcat([vec(x) for x in innovationAll]...)

    # 2a) compute MSE
    mse_reg = Statistics.mean(all_reg .^ 2)

    # 2b) compute MAE
    mae_reg = Statistics.mean(abs.(all_reg))

    return mse_reg, mae_reg    
end

function write_results(
    filename::AbstractString,
    fAll_NM, zPredNMAll, innovationAll_NM, innovation_likelihood_NM, times_NM, alloc_NM, iters_NM,
    Σw_NM,  Σv_NM,  a0_NM,  Σx_NM,  θF_NM,
    fAll_EM, zPredEMAll, innovationAll_EM, innovation_likelihood_EM, times_EM, alloc_EM, iters_EM,
    Σw_EM,  Σv_EM,  a0_EM,  Σx_EM,  θF_EM,
    zPredRKFAll, innovationAll_RKF, logLikelihoodAll_RKF,
    times, θg, firstIndex, lastIndex
)
    # helper: wrap every element (scalar or vector) into a column vector
    function cellify(x)
        cells = Vector{Any}(undef, length(x))
        for (i,v) in enumerate(x)
            arr = v isa AbstractArray ? v : [v]    # wrap scalars
            cells[i] = reshape(arr, :, 1)          # make it a column
        end
        return cells
    end

    matopen(filename, "w") do f
        write(f, "fAll_NM", fAll_NM)
        write(f, "fAll_EM", fAll_EM)

        write(f, "zPredNMAll",               cellify(zPredNMAll))
        write(f, "innovationAll_NM",         cellify(innovationAll_NM))
        write(f, "innovation_likelihood_NM", cellify(innovation_likelihood_NM))
        write(f, "times_NM", times_NM)
        write(f, "alloc_NM", alloc_NM)
        write(f, "iters_NM", iters_NM)

        write(f, "Sigma_w_NM", Σw_NM)
        write(f, "Sigma_v_NM", Σv_NM)
        write(f, "Sigma_x_NM", Σx_NM)
        write(f, "a0_NM", a0_NM)
        write(f, "theta_F_NM", θF_NM)

        write(f, "zPredEMAll",               cellify(zPredEMAll))
        write(f, "innovationAll_EM",         cellify(innovationAll_EM))
        write(f, "innovation_likelihood_EM", cellify(innovation_likelihood_EM))
        write(f, "times_EM", times_EM)
        write(f, "alloc_EM", alloc_EM)
        write(f, "iters_EM", iters_EM)

        write(f, "Sigma_w_EM", Σw_EM)
        write(f, "Sigma_v_EM", Σv_EM)
        write(f, "Sigma_x_EM", Σx_EM)
        write(f, "a0_EM", a0_EM)
        write(f, "theta_F_EM", θF_EM)

        write(f, "zPredRKFAll",              cellify(zPredRKFAll))
        write(f, "innovationAll_RKF",        cellify(innovationAll_RKF))
        write(f, "innovation_likelihood_RKF", cellify(logLikelihoodAll_RKF))

        write(f, "times",                    cellify(times))
        write(f, "theta_g", θg)
        write(f, "firstIndex", firstIndex)
        write(f, "lastIndex", lastIndex)
    end
end

end # module
