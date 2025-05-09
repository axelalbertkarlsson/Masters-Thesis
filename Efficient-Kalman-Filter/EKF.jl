#EKF.jl
module EKF

using LinearAlgebra, ProgressMeter, Dates, Optim, Printf, LineSearches, Zygote
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
# Ensure covariance stays SPD
function symmetrize_and_jitter(Σ::AbstractMatrix{T}) where T<:Real
    Σs = 0.5*(Σ + Σ')
    δ  = max(eps(T)*tr(Σs), T(1e-6))
    return Σs + δ*I(size(Σs,1))
end

# Safe innovation covariance
function safeS(H, P, R)
    S = 0.5*(H*P*H' + R + (H*P*H' + R)')
    δ = 1e-8*tr(S)
    return S + δ*I(size(S,1))
end

# ———————————————————————————————————————————————————————————————————————
# Unpack ψ → (a0, Σx, Σw, Σv, θF, θg)
function psi_to_parameters(ψ, θg_bool,
                           len_a0, n_x, m_w, p_v, r_g, len_F,
                           Σx_0, Σw_0, Σv_0, θg_0)

    idx = 1
    Treal = eltype(ψ)

    # a0
    a0 = ψ[idx:idx+len_a0-1]; idx += len_a0

    # Σx
    Lx = zeros(Treal, n_x, n_x)
    for i in 1:n_x
        Lx[i,i] = exp(ψ[idx]); idx += 1
    end
    for j in 1:n_x, i in j+1:n_x
        Lx[i,j] = ψ[idx]; idx += 1
    end
    Σx = symmetrize_and_jitter(Lx*Lx'); @assert isposdef(Σx)

    # Σw
    Lw = zeros(Treal, m_w, m_w)
    for i in 1:m_w
        Lw[i,i] = exp(ψ[idx]); idx += 1
    end
    for j in 1:m_w, i in j+1:m_w
        Lw[i,j] = ψ[idx]; idx += 1
    end
    Σw = symmetrize_and_jitter(Lw*Lw'); @assert isposdef(Σw)

    # Σv
    Lv = zeros(Treal, p_v, p_v)
    for i in 1:p_v
        Lv[i,i] = exp(ψ[idx]); idx += 1
    end
    for j in 1:p_v, i in j+1:p_v
        Lv[i,j] = ψ[idx]; idx += 1
    end
    Σv = symmetrize_and_jitter(Lv*Lv'); @assert isposdef(Σv)

    # θg
    θg = θg_bool ? reshape(ψ[idx:idx+r_g-1], size(θg_0)) : θg_0
    idx += r_g

    # θF
    logθF = ψ[idx:idx+len_F-1]
    θF    = exp.(logθF)

    return a0, Σx, Σw, Σv, θF, θg
end

# compute log‐det and Mahalanobis in pure Julia
function logdet_and_maha_pure(L, ε)
    n = size(L,1)
 
    # 1) log‐det = 2 * sum(log diag(L))
    ld = zero(eltype(L))
    for i in 1:n
      ld += 2 * log(L[i,i])
    end

    # 2) forward substitution: L * y = ε
    y = similar(ε)
    for i in 1:n
      acc = zero(eltype(L))
      for j in 1:i-1
        acc += L[i,j] * y[j]
      end
      y[i] = (ε[i] - acc) / L[i,i]
    end

    # 3) back substitution: L' * x = y
    x = similar(y)
    for i in n:-1:1
      acc = zero(eltype(L))
      for j in i+1:n
        acc += L[j,i] * x[j]
      end
      x[i] = (y[i] - acc) / L[i,i]
    end

    # 4) Mahalanobis = dot(x,x)
    mah = zero(eltype(L))
    for i in 1:n
      mah += x[i] * x[i]
    end

    return ld, mah
end

# ———————————————————————————————————————————————————————————————————————
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


# ———————————————————————————————————————————————————————————————————————
function NM(
    zAll, oIndAll, tcAll, I_z_t, f_t,
    n_c, n_p, n_s, n_t, n_u, n_x, n_z_t,
    AAll, BAll, DAll, GAll,
    firstDates, tradeDates, ecbRatechangeDates, T0All, TAll,
    a0_0, Σx_0, Σw_0, Σv_0, θF_0, θg_0;
    tol::Real=1e-6, maxiter::Int=10, verbose::Bool=false,
    θg_bool::Bool=false, chooser::Int=4,
    segmented::Bool=false,
    chunks::Vector{UnitRange{Int}} = [1:n_t]
)
    # pack initial ψ₀
    n_x  = size(Σx_0,1)
    m_w  = size(Σw_0,1)
    p_v  = size(Σv_0,1)
    r_g  = θg_bool ? length(vec(θg_0)) : 0
    len_a0 = length(a0_0)
    len_F  = length(θF_0)
    len_Lx = n_x*(n_x+1)÷2
    len_Lw = m_w*(m_w+1)÷2
    len_Lv = p_v*(p_v+1)÷2

    ψ0 = Float64[]
    append!(ψ0, vec(a0_0))
    function push_chol!(M,n)
        L = cholesky(M).L
        for i in 1:n; push!(ψ0, log(L[i,i])); end
        for j in 1:n, i in j+1:n; push!(ψ0, L[i,j]); end
    end
    push_chol!(Σx_0,n_x)
    push_chol!(Σw_0,m_w)
    push_chol!(Σv_0,p_v)
    θg_bool && append!(ψ0, vec(θg_0))
    append!(ψ0, vec(log.(θF_0 .+ 1e-6)))

    # define the chunked objective
    function fobj(ψ)
        a0, Σx, Σw, Σv, θF, θg = psi_to_parameters(
            ψ, θg_bool,
            len_a0, n_x, m_w, p_v, r_g, len_F,
            Σx_0, Σw_0, Σv_0, θg_0
        )
        neg2ℓ = zero(eltype(ψ))

        @showprogress for idxs in chunks
            x_filt = a0
            P_filt = Σx
            for (tloc,t) in enumerate(idxs)
                # build and predict
                AθFt = AAll[t] .* reshape(θF,1,:)
                Fmat = AθFt * BAll[t]
                if tloc == 1
                    x_pred = Fmat * a0
                    P_pred = Fmat*Σx*Fmat' + DAll[t]*Σw*DAll[t]'
                else
                    x_pred = Fmat * x_filt
                    P_pred = Fmat*P_filt*Fmat' + DAll[t]*Σw*DAll[t]'
                end

                # linearize
                o, _ = pricingFunctions.calcO(
                    firstDates[t], tradeDates[t],
                    θg, ecbRatechangeDates,
                    n_c, n_z_t[t],
                    T0All[t], TAll[t]
                )
                H, u, _, _ = pricingFunctions.taylorApprox(
                    o, oIndAll[t], tcAll[t],
                    x_pred[1:n_s], I_z_t[t], n_z_t[t]
                )
                R = GAll[t]*Σv*GAll[t]'
                ε = vec(zAll[t]) - H*x_pred - u

                # safe covariance + pure‐Julia logdet/maha
                S = safeS(H, P_pred, R)
                @assert isposdef(S) "S not SPD at t=$t"
                Fchol = cholesky(S)
                ld, mah = logdet_and_maha_pure(Fchol.L, ε)
                neg2ℓ += ld + mah

                # update
                K = P_pred*H'*inv(S)
                x_filt = x_pred + K*ε
                P_filt = (I - K*H)*P_pred
            end
        end

        return 0.5*neg2ℓ
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
            # 1) build & compile the tape on our real objective
            @time tape = ReverseDiff.GradientTape(fobj, ψ0)
            @info "Compiling ReverseDiff tape…"
            @showprogress for _ in 1:1
                ReverseDiff.compile(tape)
            end

            g = zeros(length(ψ0))
            @info "Computing initial gradient…"
            @showprogress for _ in 1:1
                ReverseDiff.gradient!(g, tape, ψ0)
            end

            @info "initial ∥∇f∥ =", norm(g)

            # 2) run Broyden (fobj itself has its own progress bar)
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
                println("   • optimizing block $name (length=$(length(blk))) …")
                elapsed = @elapsed optimize_block!(ψ, blk, name)
                # format time with Printf instead of round(...)
                fmt_time = @sprintf("%.3f", elapsed)
                movement = norm(ψ[blk] .- old)
                println("     → $name done in $fmt_time s; movement $movement")
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
            #ψ_opt = newtonOptimizeOptim(fobj, ψ0; tol=tol, maxiter=maxiter)
            ψ_opt = newtonOptimizeOptim_forward(fobj, ψ0; tol=tol, maxiter=maxiter, verbose=verbose)
        elseif chooser == 3
            ψ_opt = newtonOptimize(fobj, ψ0; tol=tol, maxiter=maxiter, verbose=verbose)
        elseif chooser == 4
            # 1) build & compile the tape on our real objective
            @time tape = ReverseDiff.GradientTape(fobj, ψ0)
            @info "Compiling ReverseDiff tape…"
            @showprogress for _ in 1:1
                ReverseDiff.compile(tape)
            end

            g = zeros(length(ψ0))
            @info "Computing initial gradient…"
            @showprogress for _ in 1:1
                ReverseDiff.gradient!(g, tape, ψ0)
            end

            @info "initial ∥∇f∥ =", norm(g)

            # 2) run Broyden (fobj itself has its own progress bar)
            ψ_opt = newtonOptimizeBroyden(fobj, ψ0;
                                            tol=tol,
                                            maxiter=maxiter,
                                            verbose=verbose)
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
