#EKF.jl
module EKF

using LinearAlgebra, Printf

include("pricingFunctions.jl")
include("newtonMethod.jl")

using .newtonMethod
using .pricingFunctions

export kalman_filter_smoother_lag1, NM, calcOutOfSample, EM

struct Q2Objective
    unpack_vecψ
    x_s
    P_s
    Hprev
    uprev
    E11
    E10
    E00
    AAll
    BAll
    DAll
    GAll
    zAll
    n_t
    θF_size
    jitter
end

function (q::Q2Objective)(vecψ)
    Σw_c, Σv_c, θF_c, θg_c = q.unpack_vecψ(vecψ)
    Q2 = zero(eltype(Σw_c))

    Rv = q.GAll[1]*Σv_c*q.GAll[1]' + q.jitter*I
    δz = q.zAll[1] .- q.Hprev[1]*q.x_s[1] .- q.uprev[1]
    HpPH = q.Hprev[1]*q.P_s[1]*q.Hprev[1]'
    L = cholesky(Symmetric(Rv))
    Q2 += 2 * sum(log, diag(L.L)) + δz'*(Rv\δz) + tr(Rv \ HpPH)

    for t in 2:q.n_t
        Rv = q.GAll[t]*Σv_c*q.GAll[t]' + q.jitter*I
        δz = q.zAll[t] .- q.Hprev[t]*q.x_s[t] .- q.uprev[t]
        HpPH = q.Hprev[t]*q.P_s[t]*q.Hprev[t]'
        Q2 += δz'*(Rv\δz) + tr(Rv \ HpPH) + 2 * sum(log, diag(cholesky(Symmetric(Rv)).L))

        Fm = q.AAll[t]*Diagonal(θF_c)*q.BAll[t]
        Rw = q.DAll[t]*Σw_c*q.DAll[t]' + q.jitter*I
        tmp = Fm * q.E00[t]
        resid = q.E11[t] - q.E10[t]*Fm' - Fm*q.E10[t]' + tmp*Fm'
        Q2 += tr(Rw \ resid) + 2 * sum(log, diag(cholesky(Symmetric(Rw)).L))
    end

    return 0.5 * Q2
end


function EM(
    zAll, oIndAll, tcAll, I_z_t, f_t,
    n_c, n_p, n_s, n_t, n_u, n_x, n_z_t,
    AAll, BAll, DAll, GAll,
    firstDates, tradeDates, ecbRatechangeDates, T0All, TAll,
    ψ0;
    maxiter::Int=20,
    tol::Float64=1e-6,
    verbose::Bool=false,
    θg_bool::Bool=false
)
    Σw, Σv, a0, Σx, θF, θg = ψ0
    jitter = 1e-8

    function make_packers(n_x, n_u, θF, θg, θg_bool)
        function pack(Σw, Σv, θF, θg)
            Lw = cholesky(Σw + jitter * I).L
            Lv = cholesky(Σv + jitter * I).L
            out = Float64[]
            for i in 1:n_x, j in 1:i push!(out, Lw[i,j]) end
            for i in 1:n_u, j in 1:i push!(out, Lv[i,j]) end
            append!(out, θF)
            θg_bool && append!(out, vec(θg))
            return out
        end

        function unpack(vecψ)
            idx = 1
            Lw = zeros(typeof(vecψ[1]), n_x, n_x)
            for i in 1:n_x, j in 1:i Lw[i,j] = vecψ[idx]; idx += 1 end
            Lv = zeros(typeof(vecψ[1]), n_u, n_u)
            for i in 1:n_u, j in 1:i Lv[i,j] = vecψ[idx]; idx += 1 end
            Σw_c = Lw * Lw' + jitter * I
            Σv_c = Lv * Lv' + jitter * I
            θF_c = vecψ[idx:idx+length(θF)-1]; idx += length(θF)
            θg_c = θg_bool ? reshape(vecψ[idx:idx+length(vec(θg))-1], size(θg)) : θg
            return Σw_c, Σv_c, θF_c, θg_c
        end

        return pack, unpack
    end

    pack, unpack_vecψ = make_packers(n_x, n_u, θF, θg, θg_bool)
    prev_Q = -Inf

    for k in 1:maxiter
        x_f, P_f, x_s, P_s, P_lag, oAll, EAll = kalman_filter_smoother_lag1(
            zAll, oIndAll, tcAll, I_z_t, f_t,
            n_c, n_p, n_s, n_t, n_u, n_x, n_z_t,
            AAll, BAll, DAll, GAll,
            Σw, Σv, a0, Σx, θF, θg,
            firstDates, tradeDates, ecbRatechangeDates, T0All, TAll
        )

        Hprev = Vector{Matrix{Float64}}(undef, n_t)
        uprev = Vector{Vector{Float64}}(undef, n_t)
        for t in 1:n_t
            Hprev[t], uprev[t], _, _ = taylorApprox(oAll[t], oIndAll[t], tcAll[t], x_s[t][1:n_s], I_z_t[t], n_z_t[t])
        end

        E11 = Vector{Matrix{Float64}}(undef, n_t)
        E10 = Vector{Matrix{Float64}}(undef, n_t)
        E00 = Vector{Matrix{Float64}}(undef, n_t)
        E11[1] = zeros(n_x,n_x); E10[1] = zeros(n_x,n_x); E00[1] = zeros(n_x,n_x)
        for t in 2:n_t
            E11[t] = P_s[t] + x_s[t]*x_s[t]'
            E10[t] = P_lag[t] + x_s[t]*x_s[t-1]'
            E00[t] = P_s[t-1] + x_s[t-1]*x_s[t-1]'
        end

        q2obj_struct = Q2Objective(unpack_vecψ, x_s, P_s, Hprev, uprev, E11, E10, E00, AAll, BAll, DAll, GAll, zAll, n_t, size(θF), jitter)

        vecψ0 = pack(Σw, Σv, θF, θg)
        
        #vecψ_opt = optimize_parameters(q2obj_struct, vecψ0; tol=1e-8, maxiter)
        vecψ_opt = optimize_parameters_forwarddiff(q2obj_struct, vecψ0; tol=1e-8, maxiter=10)

        Σw, Σv, θF, θg = unpack_vecψ(vecψ_opt)

        a0 = x_s[1]
        Σx = P_s[1]

        Q1 = n_x*log(2π) + logdet(Σx) + tr(inv(Σx)*(P_s[1] + (x_s[1]-a0)*(x_s[1]-a0)'))
        Q2 = 2 * q2obj_struct(pack(Σw, Σv, θF, θg))
        Qtot = Q1 + Q2

        if verbose
            @printf(" EM iter %2d: Q = %.6e   ΔQ = %.3e\n", k, Qtot, Qtot - prev_Q)
        end
        if k > 1 && abs(Qtot - prev_Q) < tol
            verbose && println(" EM converged.")
            break
        end
        prev_Q = Qtot
    end

    return kalman_filter_smoother_lag1(
        zAll, oIndAll, tcAll, I_z_t, f_t,
        n_c, n_p, n_s, n_t, n_u, n_x, n_z_t,
        AAll, BAll, DAll, GAll,
        Σw, Σv, a0, Σx, θF, θg,
        firstDates, tradeDates, ecbRatechangeDates, T0All, TAll
    )..., a0, Σx, Σw, Σv, θF, θg
end





function calcOutOfSample(zAll, oIndAll, tcAll, I_z_t, n_c, n_s, n_t, n_x, n_z_t, AAll, BAll, DAll, GAll, Σw, Σv, a0, Σx, θF, θg, firstDates, tradeDates, ecbRatechangeDates, T0All,TAll)
    T = n_t;

    # Preallocate
    x_pred = [zeros(n_x) for _ in 1:T]
    P_pred = [zeros(n_x,n_x) for _ in 1:T]
    x_filt = [zeros(n_x) for _ in 1:T]
    P_filt = [zeros(n_x,n_x) for _ in 1:T]
    K      = [zeros(n_x,n_x) for _ in 1:T]

    oAll = [zeros(103, 22) for _ in 1:T]
    EAll = [zeros(3661, 6) for _ in 1:T]

    neg2ℓ_t = zeros(Float64, T) 
    ε_t = Vector{Vector{Float64}}(undef, T) 
    zPred_t = Vector{Vector{Float64}}(undef, T) 

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
        K[t] = P_pred[t]*H_t' * inv(H_t*P_pred[t]*H_t' + R_t)
        S_t = H_t*P_pred[t]*H_t' + R_t
        ε = vec(zAll[t]) - H_t*x_pred[t] - u_t
        neg2ℓ_t[t] = logdet(S_t) + dot(ε, S_t\ε)
        ε_t[t] = ε
        zPred_t[t] = H_t*x_pred[t] + u_t
    end
        return zPred_t, ε_t, neg2ℓ_t
end

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
     Σx_0 = Symmetric((Σx_0 + Σx_0')/2)
     Σw_0 = Symmetric((Σw_0 + Σw_0')/2)
     Σv_0 = Symmetric((Σv_0 + Σv_0')/2)
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
        len_Σx = length(Σx_0)
        len_Σw = length(Σw_0)
        len_Σv = length(Σv_0)
        len_g  = length(vec(θg_0))
        len_F  = length(θF_0)
        shape_g = size(θg_0)
    else
        len_a0 = length(a0_0)
        len_Σx = length(Σx_0)
        len_Σw = length(Σw_0)
        len_Σv = length(Σv_0)
        len_F  = length(θF_0)
    end

    

    # unpack helper
    function psi_to_parameters(ψ, θg_bool)
        idx = 1
      
        # initial state
        a0 = ψ[idx:idx+len_a0-1];                     idx += len_a0
      
        # Σx, Σw, Σv via their Cholesky factors
        Lx_flat = ψ[idx:idx+len_Σx-1]; idx += len_Σx
        Lw_flat = ψ[idx:idx+len_Σw-1]; idx += len_Σw
        Lv_flat = ψ[idx:idx+len_Σv-1]; idx += len_Σv
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
      
    λ = 1e1
    μ = 1e-3
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