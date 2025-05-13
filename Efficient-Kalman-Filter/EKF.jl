#EKF.jl
module EKF

using LinearAlgebra

include("pricingFunctions.jl")
include("newtonMethod.jl")

using .newtonMethod
using .pricingFunctions

export kalman_filter_smoother_lag1, NM

function kalman_filter_smoother_lag1(zAll, oIndAll, tcAll, I_z_t, f_t, n_c, n_p, n_s, n_t, n_u, n_x, n_z_t, AAll, BAll, DAll, GAll, Σw, Σv, a0, Σx, θF, θg, firstDates, tradeDates, ecbRatechangeDates, T0All,TAll)
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
        H_t, u_t, g, Gradient = pricingFunctions.taylorApprox(oAll[t], oIndAll[t], tcAll[t], x_pred[t][1:n_s], I_z_t[t], n_z_t[t])
        R_t = GAll[t]*Σv*GAll[t]'
        jitter = 1e-8
        K[t] = P_pred[t]*H_t' * inv(H_t*P_pred[t]*H_t' + R_t + jitter*I)
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
        jitter = 1e-8
        S[t] = P_filt[t]*(AAll[t+1]*Diagonal(θF)*BAll[t+1])' * inv(P_pred[t+1] + jitter*I)

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

function NM(
    zAll, oIndAll, tcAll, I_z_t, f_t,
    n_c, n_p, n_s, n_t, n_u, n_x, n_z_t,
    AAll, BAll, DAll, GAll,
    firstDates, tradeDates, ecbRatechangeDates, T0All, TAll,
    a0_0::AbstractArray{<:AbstractFloat},
    Σx_0::AbstractArray{<:AbstractFloat},
    Σw_0::AbstractArray{<:AbstractFloat},
    Σv_0::AbstractArray{<:AbstractFloat},
    θF_0::AbstractArray{<:AbstractFloat},
    θg_0::AbstractArray{<:AbstractFloat};
    tol::Float64=1e-6,
    maxiter::Int=10,
    verbose::Bool=false,
    Newton_bool::Bool=false,
    θg_bool::Bool=false,  
)
     # flatten ψ₀ with unconstrained params:
     #  - we store lower‐triangular entries of chol(Σ) (so PD is automatic)
     #  - we store unconstrained φ_F such that θ_F = tanh(φ_F)
     Lx0 = cholesky(Σx_0).L
     Lw0 = cholesky(Σw_0).L
     Lv0 = cholesky(Σv_0).L
     #φF0 = atanh.(θF_0)    # inverse of tanh
  # make sure θF_0 is strictly inside (–1,1) so atanh never returns ±Inf
        θF0_safe = clamp.(θF_0, -1 + 1e-6, 1 - 1e-6)
        φF0      = atanh.(θF0_safe)


     if θg_bool
         ψ0 = vcat(
           vec(a0_0),
           vec(tril(Lx0)),       # lower‐tri entries
           vec(tril(Lw0)),
           vec(tril(Lv0)),
           vec(θg_0),
           φF0
         )
     else
         ψ0 = vcat(
        vec(a0_0),
        vec(tril(Lx0)),
        vec(tril(Lw0)),
        vec(tril(Lv0)),
        φF0
    )
    end
    
    
    # chunk lengths
    if (θg_bool)
        len_a0 = length(a0_0)
        len_Sx = length(Σx_0)
        len_Sw = length(Σw_0)
        len_Sv = length(Σv_0)
        len_g  = length(vec(θg_0))
        len_F  = length(θF_0)
        shape_g = size(θg_0)
    else
        len_a0 = length(a0_0)
        len_Sx = length(Σx_0)
        len_Sw = length(Σw_0)
        len_Sv = length(Σv_0)
        len_F  = length(θF_0)
    end

    

    # unpack helper
    function psi_to_parameters(ψ, θg_bool)
        idx = 1
      
        # initial state
        a0 = ψ[idx:idx+len_a0-1];                     idx += len_a0
      
        # Σx, Σw, Σv via their Cholesky factors
        Lx_flat = ψ[idx:idx+len_Sx-1]; idx += len_Sx
        Lw_flat = ψ[idx:idx+len_Sw-1]; idx += len_Sw
        Lv_flat = ψ[idx:idx+len_Sv-1]; idx += len_Sv
        Lx = reshape(Lx_flat, size(Σx_0))
        Lw = reshape(Lw_flat, size(Σw_0))
        Lv = reshape(Lv_flat, size(Σv_0))
        Σx = Lx * Lx'   # guaranteed PD
        Σw = Lw * Lw'
        Σv = Lv * Lv'
      
        # θg if requested
        if θg_bool
          θg_flat = ψ[idx:idx+len_g-1];                idx += len_g
          θg      = reshape(θg_flat, shape_g)
        else
          θg = θg_0
        end
      
        # finally φF → θF via tanh
        φF = ψ[idx:idx+len_F-1]
        φF = clamp.(φF, atanh(-1+1e-6), atanh(1-1e-6))
        θF = tanh.(φF)
      
        return a0, Σx, Σw, Σv, θF, θg
      end
      
    λ = 1e2
    μ = 1e-4
    # objective: negative log‐likelihood (filter only)
    fobj = function(ψ)
        try
        a0, Σx, Σw, Σv, θF, θg = psi_to_parameters(ψ, θg_bool)
        # allow tracked arrays by using Any
        x_pred     = Vector{Any}(undef, n_t)
        P_pred     = Vector{Any}(undef, n_t)
        x_filt_loc = Vector{Any}(undef, n_t)
        P_filt_loc = Vector{Any}(undef, n_t)
        neg2ℓ = 0.0

        # t=1
        x_pred[1] = AAll[1] * (θF .* (BAll[1] * a0))
        K0 = BAll[1] * Σx * BAll[1]'
        WeightedK0 = K0 .* (θF * θF')
        P_pred[1] = AAll[1] * WeightedK0 * AAll[1]' + DAll[1] * Σw * DAll[1]'

        o1,_ = calcO(firstDates[1], tradeDates[1], θg, ecbRatechangeDates, n_c, n_z_t[1], T0All[1], TAll[1])
        H1,u1,_,_ = taylorApprox(o1, oIndAll[1], tcAll[1], x_pred[1][1:n_s], I_z_t[1], n_z_t[1])
        R1 = GAll[1]*Σv*GAll[1]'
        S1 = H1*P_pred[1]*H1' + R1
        ε1 = vec(zAll[1]) - H1*x_pred[1] - u1
        neg2ℓ += logdet(S1) + dot(ε1, S1\ε1)
        Kmat = P_pred[1]*H1'*inv(S1)
        x_filt_loc[1] = x_pred[1] + Kmat*ε1
        P_filt_loc[1] = (I - Kmat*H1)*P_pred[1]

        # t=2:T
        for t in 2:n_t
            x_pred[t] = AAll[t] * (θF .* (BAll[t] * x_filt_loc[t-1]))
            Kt = BAll[t] * P_filt_loc[t-1] * BAll[t]'
            Wt = Kt .* (θF * θF')
            P_pred[t] = AAll[t] * Wt * AAll[t]' + DAll[t] * Σw * DAll[t]'
            o_t,_ = calcO(firstDates[t], tradeDates[t], θg, ecbRatechangeDates, n_c, n_z_t[t], T0All[t], TAll[t])
            H_t,u_t,_,_ = taylorApprox(o_t, oIndAll[t], tcAll[t], x_pred[t][1:n_s], I_z_t[t], n_z_t[t])
            R_t = GAll[t]*Σv*GAll[t]'
            S_t = H_t*P_pred[t]*H_t' + R_t
            ε = vec(zAll[t]) - H_t*x_pred[t] - u_t
            neg2ℓ += logdet(S_t) + dot(ε, S_t\ε)
            Kmat = P_pred[t]*H_t'*inv(S_t)
            x_filt_loc[t] = x_pred[t] + Kmat*ε
            P_filt_loc[t] = (I - Kmat*H_t)*P_pred[t]
        end

        #return 0.5 * neg2ℓ
        # penalize large x_pred norms (across t=1:T)
            pen = sum(norm.(x_pred).^2)
        
            return 0.5*neg2ℓ + μ*pen + λ*sum(abs2, ψ)
        catch
            return 1e8 + λ*sum(abs2, ψ)
      end
    end

    # optimize
    if (Newton_bool)
        ψ_opt = newtonOptimize(fobj, ψ0; tol=tol, maxiter=maxiter, verbose=verbose)
    else
        #ψ_opt = newtonOptimizeBroyden(fobj, ψ0; tol=tol, maxiter=maxiter, verbose=verbose)
        ψ_opt = optimize_parameters(fobj, ψ0;
                                tol=tol,
                                maxiter=maxiter,
                                verbose=verbose
                            )
        # ψ_opt = optimize_bfgs(fobj, ψ0;
        #         tol=tol,
        #         maxiter=maxiter,
        #         verbose=verbose)
    end
    # unpack
    a0_opt, Σx_opt, Σw_opt, Σv_opt, θF_opt, θg_opt = psi_to_parameters(ψ_opt, θg_bool)

    # run full smoother
    x_filt, P_filt, x_smooth, P_smooth, P_lag, oAll, EAll =
        kalman_filter_smoother_lag1(
            zAll, oIndAll, tcAll, I_z_t, f_t,
            n_c, n_p, n_s, n_t, n_u, n_x, n_z_t,
            AAll, BAll, DAll, GAll,
            Σw_opt, Σv_opt, a0_opt, Σx_opt, θF_opt, θg_opt,
            firstDates, tradeDates, ecbRatechangeDates, T0All, TAll
        )

    return x_filt, P_filt, x_smooth, P_smooth, P_lag, oAll, EAll,
           a0_opt, Σx_opt, Σw_opt, Σv_opt, θF_opt, θg_opt
end

end # module