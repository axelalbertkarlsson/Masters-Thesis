#EKF.jl
module EKF

using LinearAlgebra, ProgressMeter, Dates, Optim, Printf
using ReverseDiff
using ForwardDiff
using LinearMaps
using IterativeSolvers: cg



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
if Treal==Float64
    λmin = minimum(eigvals(Σx))
    @assert λmin > 1e-12 "Σx nearly singular (min eig = $λmin)"
end

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
if Treal==Float64
    λmin = minimum(eigvals(Σw))
    @assert λmin > 1e-12 "Σw nearly singular (min eig = $λmin)"
end

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
if Treal==Float64
    λmin = minimum(eigvals(Σv))
    @assert λmin > 1e-12 "Σv nearly singular (min eig = $λmin)"
end


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
if Treal==Float64
    θmin = minimum(θF)
    @assert θmin > 1e-6 "θF entries too small (min θF = $θmin)"
end

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

function _safe_solve(S, ε)
    try
        return S \ ε
    catch err
        @warn "direct solve failed, falling back to pinv" exception=err
        return pinv(Matrix(S)) * ε
    end
 end

function safeS(H, P, R)
    S = 0.5*(H*P*H' + R + (H*P*H' + R)')   # symmetrize in one go
    #δ = eps(eltype(S)) * tr(S)
    δ = max(eps(eltype(S)) * tr(S), one(eltype(S))*1e-6)
    return S + δ * I(size(S,1))
end


function NM(
    zAll, oIndAll, tcAll, I_z_t, f_t,
    n_c, n_p, n_s, n_t, n_u, n_x, n_z_t,
    AAll, BAll, DAll, GAll,
    firstDates, tradeDates, ecbRatechangeDates, T0All, TAll,
    a0_0, Σx_0, Σw_0, Σv_0, θF_0, θg_0;
    tol::Real=1e-6, maxiter::Int=10, verbose::Bool=false,
    θg_bool::Bool=false, chooser::Int=1, segmented::Bool=false
)

    # — dimension constants & pack initial ψ₀ via Cholesky reparam —
    n_x    = size(Σx_0,1)
    m_w    = size(Σw_0,1)
    p_v    = size(Σv_0,1)
    r_g    = θg_bool ? length(vec(θg_0)) : 0
    len_a0 = length(a0_0)
    len_F  = length(θF_0)
    len_Lx = n_x*(n_x+1) ÷ 2
    len_Lw = m_w*(m_w+1) ÷ 2
    len_Lv = p_v*(p_v+1) ÷ 2

    println("Expected ψ length = $(len_a0 + len_Lx + len_Lw + len_Lv + r_g + len_F)")

    ψ0 = Float64[]
    append!(ψ0, vec(a0_0))
    function push_chol!(M, n)
        L = cholesky(M).L
        for i in 1:n
            push!(ψ0, log(L[i,i]))
        end
        for j in 1:n, i in (j+1):n
            push!(ψ0, L[i,j])
        end
    end
    push_chol!(Σx_0, n_x)
    push_chol!(Σw_0, m_w)
    push_chol!(Σv_0, p_v)
    θg_bool && append!(ψ0, vec(θg_0))
    append!(ψ0, vec(log.(θF_0 .+ 1e-6)))
    println("Initial ψ₀ length = $(length(ψ0))")

    # — allocate filtering buffers once —
    x_pred     = Vector{Any}(undef, n_t)
    P_pred     = Vector{Any}(undef, n_t)
    x_filt_loc = Vector{Any}(undef, n_t)
    P_filt_loc = Vector{Any}(undef, n_t)

    # — the negative‐log‐likelihood objective —
    fobj = function(ψ)
        FloatEl = Float64
        isdual = !(eltype(ψ) <: FloatEl)

        # unpack ψ → parameters
        a0, Σx, Σw, Σv, θF, θg = psi_to_parameters(
            ψ, θg_bool, len_a0, n_x, m_w, p_v, r_g, len_F,
            Σx_0, Σw_0, Σv_0, θg_0
        )

        neg2ℓ = zero(eltype(ψ))

        # — t = 1 —
        pred1     = θF .* (BAll[1] * a0)
        x_pred[1] = AAll[1] * pred1

        W0        = (BAll[1] * Σx * BAll[1]') .* (θF * θF')
        P_pred[1] = AAll[1] * W0 * AAll[1]' +
                    DAll[1] * Σw * DAll[1]'

        o1, _     = calcO(
            firstDates[1], tradeDates[1], θg, ecbRatechangeDates,
            n_c, n_z_t[1], T0All[1], TAll[1]
        )
        H1, u1, _, _ = taylorApprox(
            o1, oIndAll[1], tcAll[1],
            x_pred[1][1:n_s], I_z_t[1], n_z_t[1]
        )

        S1        = safeS(H1, P_pred[1], GAll[1] * Σv * GAll[1]')
        ε1        = vec(zAll[1]) .- H1 * x_pred[1] .- u1
        neg2ℓ    += logdet(S1) + dot(ε1, S1 \ ε1)

        x_filt_loc[1] = x_pred[1] + (P_pred[1] * H1') * (S1 \ ε1)
        P_filt_loc[1] = (I - (P_pred[1] * H1') * (S1 \ H1)) * P_pred[1]

        # — t = 2:n_t with prints & progress bar —
        # if !isdual
        #     println("\n→ Starting filtering loop for t = 2:$n_t")
        # end
        last_print = time()

        @showprogress for t in 2:n_t
            if !isdual && (time() - last_print > 2.0)
                println("\n→ t = $t   partial neg2ℓ = ", neg2ℓ/2)
                last_print = time()
            end

            # predict
            pred_t     = θF .* (BAll[t] * x_filt_loc[t-1])
            x_pred[t]  = AAll[t] * pred_t

            # covariance predict
            tmp        = BAll[t] * P_filt_loc[t-1] * BAll[t]'
            Wt         = tmp .* (θF * θF')
            P_pred[t]  = AAll[t] * Wt * AAll[t]' +
                         DAll[t] * Σw * DAll[t]'

            # update
            o_t, _     = calcO(
                firstDates[t], tradeDates[t], θg,
                ecbRatechangeDates, n_c, n_z_t[t], T0All[t], TAll[t]
            )
            H_t, u_t, _, _ = taylorApprox(
                o_t, oIndAll[t], tcAll[t],
                x_pred[t][1:n_s], I_z_t[t], n_z_t[t]
            )
            S_t        = safeS(H_t, P_pred[t], GAll[t] * Σv * GAll[t]')
            ε_t        = vec(zAll[t]) .- H_t * x_pred[t] .- u_t
            neg2ℓ    += logdet(S_t) + dot(ε_t, S_t \ ε_t)

            x_filt_loc[t] = x_pred[t] + (P_pred[t] * H_t') * (S_t \ ε_t)
            P_filt_loc[t] = (I - (P_pred[t] * H_t') * (S_t \ H_t)) * P_pred[t]
        end

        # if !isdual
        #     println("\n→ Exiting fobj with neg2ℓ = ", 0.5 * neg2ℓ)
        # end
        return 0.5 * neg2ℓ
    end

    # — build block ranges in ψ — 
    idx    = 1
    a0_idx = idx:(idx+len_a0-1);    idx += len_a0
    Lx_idx = idx:(idx+len_Lx-1);    idx += len_Lx
    Lw_idx = idx:(idx+len_Lw-1);    idx += len_Lw
    Lv_idx = idx:(idx+len_Lv-1);    idx += len_Lv
    θg_idx = θg_bool ? (idx:(idx+r_g-1)) : Int[];  idx += θg_bool ? r_g : 0
    θF_idx = idx:(idx+len_F-1)

    blocks = [a0_idx, Lx_idx, Lw_idx, Lv_idx]
    θg_bool && push!(blocks, θg_idx)
    push!(blocks, θF_idx)

    # names for each block, in the same order
    block_names = ["a0", "Lx", "Lw", "Lv"]
    θg_bool && push!(block_names, "θg")
    push!(block_names, "θF")

    function optimize_block!(ψf::Vector{Float64}, idxs, block_name)
        x0 = ψf[idxs]
        block_obj = x -> begin
            T = typeof(x[1])
            tmp = convert.(T, ψf)
            tmp[idxs] = x
            return fobj(tmp)
        end

        if chooser == 1
            @time ψ_block = newton_krylov(block_obj, x0;
                                    tol=tol,
                                    maxiter=maxiter,
                                    cg_tol=1e-2,
                                    cg_maxiter=maxiter)
        elseif chooser == 2
            @time ψ_block = newtonOptimizeOptim(block_obj, x0; tol=tol, maxiter=maxiter)
        elseif chooser == 3
            @time ψ_block = newtonOptimize(block_obj, x0; tol=tol, maxiter=maxiter, verbose=verbose)
        elseif chooser == 4
            @time ψ_block = newtonOptimizeBroyden(block_obj, x0; tol=tol, maxiter=maxiter, verbose=verbose)
        elseif chooser == 5
            @time ψ_block = newtonOptimizeHF(block_obj, x0; tol=tol, maxiter=maxiter, verbose=verbose)
        end

        ψf[idxs] = Float64.(ψ_block)
    end

    if segmented
        ψ = copy(ψ0)
        println("⇢ segmented block optimization (chooser=$chooser)")
        for pass in 1:maxiter
            total_move = 0.0
            println("→ pass $pass")
            for (blk, name) in zip(blocks, block_names)
                old = ψ[blk]
                println("   • optimizing block `$name` (length=$(length(blk))) …")
                elapsed = @elapsed optimize_block!(ψ, blk, name)
                # format time with Printf instead of round(...)
                fmt_time = @sprintf("%.3f", elapsed)
                movement = norm(ψ[blk] .- old)
                println("     → `$name` done in $fmt_time s; movement $movement")
                total_move += movement
            end
            println("→ total movement = $total_move")
            if total_move < tol
                println("→ converged after $pass passes")
                break
            end
        end
        ψ_opt = ψ
    else
        println("⇢ full‐vector optimizer chooser=$chooser")
        if chooser == 1
            ψ_opt = newton_krylov(fobj, ψ0; tol=tol, maxiter=maxiter,
                                  cg_tol=1e-2, cg_maxiter=min(50,length(ψ0)))
        elseif chooser == 2
            ψ_opt = newtonOptimizeOptim(fobj, ψ0; tol=tol, maxiter=maxiter)
        elseif chooser == 3
            ψ_opt = newtonOptimize(fobj, ψ0; tol=tol, maxiter=maxiter, verbose=verbose)
        elseif chooser == 4
            ψ_opt = newtonOptimizeBroyden(fobj, ψ0; tol=tol, maxiter=maxiter, verbose=verbose)
        elseif chooser == 5
            ψ_opt = newtonOptimizeHF(fobj, ψ0; tol=tol, maxiter=maxiter, verbose=verbose)
        else
            error("Unknown chooser: $chooser")
        end
    end

    # — final unpack & smoothing —
    a0_opt, Σx_opt, Σw_opt, Σv_opt, θF_opt, θg_opt = psi_to_parameters(
        ψ_opt, θg_bool, len_a0, n_x, m_w, p_v, r_g, len_F,
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





end # module EKF
