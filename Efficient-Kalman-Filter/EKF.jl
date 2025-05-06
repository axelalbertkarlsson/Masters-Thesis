#EKF.jl
module EKF

using LinearAlgebra

include("pricingFunctions.jl")
include("newtonMethod.jl")

using .newtonMethod
using .pricingFunctions

export kalman_filter_smoother_lag1, NM

# ———————————————————————————————————————————————————————————————————————
# Reparameterized pack / unpack for ψ → (a0, Σx, Σw, Σv, θF, θg)
# ensures all covariances remain SPD under small perturbations.

# These length/shape constants must be defined before NM is called.
# You can compute them just inside NM before packing ψ₀.
# For clarity they appear here as comments:

# n_x  = size(Σx_0,1)
# m_w  = size(Σw_0,1)
# p_v  = size(Σv_0,1)
# r_g  = θg_bool ? length(vec(θg_0)) : 0
# len_a0 = length(a0_0)
# len_Lx  = n_x*(n_x+1) ÷ 2
# len_Lw  = m_w*(m_w+1) ÷ 2
# len_Lv  = p_v*(p_v+1) ÷ 2
# len_F   = length(θF_0)

"""
    symmetrize_and_jitter(Σ::AbstractMatrix{T}) where T<:Real

Given Σ = L * L', enforce exact symmetry then add a tiny
Tikhonov jitter δ·I so that Σ is numerically SPD.
"""
function symmetrize_and_jitter(Σ::AbstractMatrix{T}) where T<:Real
    # 1) exact symmetry
    Σs = 0.5*(Σ + Σ')
    # 2) adaptive jitter at least 1e-12, or eps·trace(Σ)
    δ = max(eps(T)*tr(Σs), T(1e-12))
    return Σs + δ*I(size(Σs,1))
end

# Unpack ψ → parameters
# ———————————————————————————————————————————————————————————————————————
# Unpack ψ → (a0, Σx, Σw, Σv, θF, θg)
function psi_to_parameters(ψ, θg_bool,
    len_a0, n_x, m_w, p_v, r_g, len_F,
    Σx_0, Σw_0, Σv_0, θg_0)
idx = 1
Treal = eltype(ψ)
if Treal == Float64 && any(!isfinite, ψ)
error("ψ has non-finite entries: ", ψ)
end

# 1) a0
a0 = ψ[idx:idx+len_a0-1]; idx += len_a0

# 2) Σx
Lx = zeros(Treal, n_x, n_x)
for i in 1:n_x
Lx[i,i] = exp(ψ[idx]); idx += 1
end
for j in 1:n_x, i in (j+1):n_x
Lx[i,j] = ψ[idx]; idx += 1
end
# enforce symmetry + jitter
Σx = symmetrize_and_jitter(Lx * Lx')
@assert isposdef(Σx)   "Σx is still not numerically SPD"
@assert all(isfinite, Σx) "Σx contains non-finite entries"

# 3) Σw
Lw = zeros(Treal, m_w, m_w)
for i in 1:m_w
Lw[i,i] = exp(ψ[idx]); idx += 1
end
for j in 1:m_w, i in (j+1):m_w
Lw[i,j] = ψ[idx]; idx += 1
end
Σw = symmetrize_and_jitter(Lw * Lw')
@assert isposdef(Σw)   "Σw is still not numerically SPD"
@assert all(isfinite, Σw) "Σw contains non-finite entries"

# 4) Σv
Lv = zeros(Treal, p_v, p_v)
for i in 1:p_v
Lv[i,i] = exp(ψ[idx]); idx += 1
end
for j in 1:p_v, i in (j+1):p_v
Lv[i,j] = ψ[idx]; idx += 1
end
Σv = symmetrize_and_jitter(Lv * Lv')
@assert isposdef(Σv)   "Σv is still not numerically SPD"
@assert all(isfinite, Σv) "Σv contains non-finite entries"

# 5) θg
if θg_bool
θg_flat = ψ[idx:idx+r_g-1]; idx += r_g
θg = reshape(θg_flat, size(θg_0))
else
θg = θg_0
end

# 6) θF
# last block of ψ is the log-θF entries
logθF = ψ[idx:idx+len_F-1]
θF    = exp.(logθF)  # back to the positive scale

return a0, Σx, Σw, Σv, θF, θg
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

function safeS(H, P, R)
    S = 0.5*(H*P*H' + R + (H*P*H' + R)')   # symmetrize in one go
    δ = eps(eltype(S)) * tr(S)
    return S + δ * I(size(S,1))
end


function NM(
    zAll, oIndAll, tcAll, I_z_t, f_t,
    n_c, n_p, n_s, n_t, n_u, n_x, n_z_t,
    AAll, BAll, DAll, GAll,
    firstDates, tradeDates, ecbRatechangeDates, T0All, TAll,
    a0_0, Σx_0, Σw_0, Σv_0, θF_0, θg_0;
    tol=1e-6, maxiter=10, verbose=false,
    Newton_bool=false, θg_bool=false, HF_bool=false
)
    # — dimension constants —
    n_x    = size(Σx_0,1)
    m_w    = size(Σw_0,1)
    p_v    = size(Σv_0,1)
    r_g    = θg_bool ? length(vec(θg_0)) : 0
    len_a0 = length(a0_0)
    len_F  = length(θF_0)

    len_Lx = n_x*(n_x+1) ÷ 2
    len_Lw = m_w*(m_w+1) ÷ 2
    len_Lv = p_v*(p_v+1) ÷ 2
    total_expected = len_a0 + len_Lx + len_Lw + len_Lv + r_g + len_F
    println("EXPECTED ψ length = $total_expected")

    # — pack initial ψ0 via Cholesky reparam —
    ψ0 = Float64[]
    append!(ψ0, vec(a0_0))
    function push_chol!(M,n)
        L = cholesky(M).L
        for i in 1:n; push!(ψ0, log(L[i,i])); end
        for j in 1:n, i in (j+1):n; push!(ψ0, L[i,j]); end
    end
    push_chol!(Σx_0, n_x)
    push_chol!(Σw_0, m_w)
    push_chol!(Σv_0, p_v)
    θg_bool && append!(ψ0, vec(θg_0))
    # replace raw θF with its log so that during optimization exp(·) always stays positive
    ε = 1e-6
    logθF₀ = log.(θF_0 .+ ε)      # avoids log(0)
    append!(ψ0, vec(logθF₀))

    println("Initial ψ₀ length = ", length(ψ0), "  (should be $total_expected)")

    # — negative log‐likelihood with local jitter —
    fobj = function(ψ)
        jitter = 1e-8
        # unpack ψ → (a0, Σx, Σw, Σv, θF, θg)
        a0, Σx, Σw, Σv, θF, θg = psi_to_parameters(
            ψ, θg_bool,
            len_a0, n_x, m_w, p_v, r_g, len_F,
            Σx_0, Σw_0, Σv_0, θg_0
        )
        neg2ℓ = 0.0
        x_pred     = Vector{Any}(undef, n_t)
        P_pred     = Vector{Any}(undef, n_t)
        x_filt_loc = Vector{Any}(undef, n_t)
        P_filt_loc = Vector{Any}(undef, n_t)

        # t = 1
        x_pred[1] = AAll[1]*(θF .* (BAll[1]*a0))
        W0 = (BAll[1]*Σx*BAll[1]') .* (θF*θF')
        P_pred[1] = AAll[1]*W0*AAll[1]' + DAll[1]*Σw*DAll[1]'
        o1,_ = calcO(firstDates[1], tradeDates[1], θg, ecbRatechangeDates,
                     n_c, n_z_t[1], T0All[1], TAll[1])
        H1,u1,_,_ = taylorApprox(o1, oIndAll[1], tcAll[1],
                                 x_pred[1][1:n_s], I_z_t[1], n_z_t[1])                                                       
        #S1 = H1*P_pred[1]*H1' + GAll[1]*Σv*GAll[1]'
        S1 = safeS(H1, P_pred[1], GAll[1]*Σv*GAll[1]')
        ε1 = vec(zAll[1]) - H1*x_pred[1] - u1
        if !isposdef(S1) || any(!isfinite, S1) || any(!isfinite, ε1)
            return Inf
        end
        neg2ℓ += logdet(S1) + dot(ε1, S1\ε1)
        x_filt_loc[1] = x_pred[1] + (P_pred[1]*H1')*(S1\ε1)
        P_filt_loc[1] = (I - (P_pred[1]*H1')*(S1\H1))*P_pred[1]

        # t = 2:T
        for t in 2:n_t
            x_pred[t] = AAll[t]*(θF .* (BAll[t]*x_filt_loc[t-1]))
            Wt = (BAll[t]*P_filt_loc[t-1]*BAll[t]') .* (θF*θF')
            P_pred[t] = AAll[t]*Wt*AAll[t]' + DAll[t]*Σw*DAll[t]'
            o_t,_ = calcO(firstDates[t], tradeDates[t], θg,
                          ecbRatechangeDates, n_c, n_z_t[t], T0All[t], TAll[t])
            H_t,u_t,_,_ = taylorApprox(o_t, oIndAll[t], tcAll[t],
                                       x_pred[t][1:n_s], I_z_t[t], n_z_t[t])
            #S_t = H_t*P_pred[t]*H_t' + GAll[t]*Σv*GAll[t]'
            S_t = safeS(H_t, P_pred[t], GAll[t]*Σv*GAll[t]')
            ε = vec(zAll[t]) - H_t*x_pred[t] - u_t

            if !isposdef(S_t) || any(!isfinite, S_t) || any(!isfinite, ε)
                return Inf
            end

            neg2ℓ += logdet(S_t) + dot(ε, S_t\ε)
            x_filt_loc[t] = x_pred[t] + (P_pred[t]*H_t')*(S_t\ε)
            P_filt_loc[t] = (I - (P_pred[t]*H_t')*(S_t\H_t))*P_pred[t]
        end

        return 0.5 * neg2ℓ
    end

    # — optimizer chooser —
    if Newton_bool && HF_bool
        ψ_opt = newtonOptimizeHF(fobj, ψ0; tol=tol, maxiter=maxiter, verbose=verbose)
    elseif Newton_bool && !HF_bool
        ψ_opt = newtonOptimize(fobj, ψ0; tol=tol, maxiter=maxiter, verbose=verbose)
    else
        ψ_opt = newtonOptimizeBroyden(fobj, ψ0; tol=tol, maxiter=maxiter, verbose=verbose)
    end

    @assert length(ψ0) == total_expected "\
ψ₀ length $(length(ψ0)) ≠ expected $total_expected" 

    # — unpack & final smoother —
    a0_opt, Σx_opt, Σw_opt, Σv_opt, θF_opt, θg_opt = psi_to_parameters(
        ψ_opt, θg_bool,
        len_a0, n_x, m_w, p_v, r_g, len_F,
        Σx_0, Σw_0, Σv_0, θg_0
    )

    return kalman_filter_smoother_lag1(
        zAll, oIndAll, tcAll, I_z_t, f_t,
        n_c,n_p,n_s,n_t,n_u,n_x,n_z_t,
        AAll,BAll,DAll,GAll,
        Σw_opt,Σv_opt,a0_opt,Σx_opt,θF_opt,θg_opt,
        firstDates,tradeDates,ecbRatechangeDates,T0All,TAll
    )..., a0_opt, Σx_opt, Σw_opt, Σv_opt, θF_opt, θg_opt
end

end # module
