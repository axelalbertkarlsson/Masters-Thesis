# newtonMethod.jl

module newtonMethod

using ReverseDiff, LinearAlgebra, LinearMaps, ForwardDiff, Optim, LineSearches, ProgressMeter, Dates, Printf
#using Optim: value, gradient, minimizer
using IterativeSolvers: cg

export newtonStep, newtonOptimize, newtonOptimizeBroyden, newtonOptimizeHF, newton_krylov, newtonOptimizeOptim

# ———————————————————————————————————————————————————————————————————————
# Full Newton (unchanged)
function newtonStep(f, grad_tape, hess_tape,
                    ψ; α=1e-4, s0=1.0, s_min=1e-8, verbose=false)
    n = length(ψ)
    g = zeros(n); ReverseDiff.gradient!(g, grad_tape, ψ)
    H = zeros(n,n); ForwardDiff.hessian!(H, hess_tape, ψ)
    Δψ = - H \ g
    f0 = f(ψ); s = s0
    while s ≥ s_min && f(ψ .+ s.*Δψ) > f0 + α*s*dot(g,Δψ)
        s *= 0.5
    end
    verbose && println("  ‖Δψ‖=",norm(Δψ)," step=",s)
    return ψ .+ s.*Δψ, norm(Δψ)
end

function newtonOptimize(f, ψ₀; tol=1e-4, maxiter=5, verbose=false)
    ψ = copy(ψ₀)
    println("-> Building Gradient tape...")
    grad_tape = ReverseDiff.GradientTape(f, ψ₀)
    println("-> Compiling Gradient tape...")
    ReverseDiff.compile(grad_tape)
    println("-> Building Hessian tape...")
    hess_tape = ReverseDiff.HessianTape(f, ψ₀)
    println("-> Compiling Hessian tape...")
    ForwardDiff.compile!(hess_tape)
    for k in 1:maxiter
        ψ_new, δ = newtonStep(f, grad_tape, hess_tape, ψ; verbose=verbose)
        ψ = ψ_new
        if δ < tol
            verbose && println("Converged at iter $k")
            break
        end
    end
    return ψ
end

# ———————————————————————————————————————————————————————————————————————
# Hessian-free Newton
"""
    newtonOptimizeHF(f, ψ₀; tol, maxiter, verbose)

Uses ReverseDiff for ∇f and finite-difference to build H·v,
then Conjugate-Gradient on the LinearMap to approximate Δψ.
"""
function newtonOptimizeHF(f, ψ₀::Vector{Float64};
    tol::Float64=1e-6,
    maxiter::Int=5,
    verbose::Bool=false
)
    ψ = copy(ψ₀)
    n = length(ψ)

    f0 = f(ψ)
    g0 = ForwardDiff.gradient(f, ψ)

    if any(!isfinite, g0)
        @warn "Initial gradient contains NaNs or Infs"
        return ψ₀
    end

    println("Sanity HF @ ψ₀ → f = $f0, ‖g‖ = $(norm(g0))")

    for k in 1:maxiter
        g = ForwardDiff.gradient(f, ψ)
        @assert all(isfinite, g) "∇f contains non‐finite entries at iteration $k: indices = $(findall(!isfinite, g))"
        if any(!isfinite, g)
            @warn "Gradient became non-finite at iter $k"
            break
        end

        ε = eps(Float64)^(1/3)
        grad_tape = ReverseDiff.GradientTape(f, ψ)
        ReverseDiff.compile(grad_tape)

        Hlin = LinearMap{Float64}((v,out)->begin
            gp = similar(out); ReverseDiff.gradient!(gp, grad_tape, ψ .+ ε .* v)
            gm = similar(out); ReverseDiff.gradient!(gm, grad_tape, ψ .- ε .* v)
            @. out = (gp - gm)/(2ε)
        end, n, n; ismutating=true)

        Δψ, _ = cg(Hlin, -g; abstol=tol, maxiter=50)

        if any(!isfinite, Δψ) || !isfinite(norm(Δψ)) || norm(Δψ) > 1e6
            @warn "HF produced invalid step (NaN or too large norm=$(norm(Δψ))); aborting HF."
            break
        end

        # Backtracking
        found_good_step = false
        s = 1.0
        base = f(ψ)
        while s > 1e-8
            candidate = ψ .+ s .* Δψ
            ftrial = f(candidate)
            if isfinite(ftrial) && ftrial < base
                ψ .= candidate
                found_good_step = true
                break
            end
            s *= 0.5
        end

        if !found_good_step
            @warn "HF line-search failed to find a finite, improving ftrial; stopping HF."
            break
        end

        verbose && println(" HF iter $k → ‖Δψ‖=$(norm(Δψ)), step=$s, f=$(f(ψ))")

        if norm(Δψ) < tol
            verbose && println(" HF converged at iter $k (‖Δψ‖=$(norm(Δψ)))")
            break
        end
    end

    return ψ
end



# ———————————————————————————————————————————————————————————————————————
# function newtonOptimizeBroyden(f, ψ₀; tol=1e-6, maxiter=10, verbose=false)
#     @info "Starting BFGS optimization" tol=tol maxiter=maxiter

#     safe_f(x) = try
#         f(x)
#     catch err
#         isa(err, DomainError) ? Inf : rethrow(err)
#     end

#     @info "Building ReverseDiff gradient tape…"
#     tape = ReverseDiff.GradientTape(safe_f, ψ₀)
#     @info "→ Done building tape"

#     @info "Compiling ReverseDiff gradient tape…"
#     ReverseDiff.compile(tape)
#     @info "→ Done compiling tape"

#     gradient! = (g,x) -> (ReverseDiff.gradient!(g, tape, x); g)

#     opts = Optim.Options(g_tol=tol,
#                          iterations=maxiter,
#                          show_trace=verbose,
#                          store_trace=verbose)

#     ## ——— CHOOSE YOUR LINE SEARCH ———
#     # 1) Strong‐Wolfe (MoreThuente) — usually bolder than backtracking
#     #method = Optim.LBFGS(linesearch = LineSearches.MoreThuente(), m = 20)

#     # 2) Relaxed back-tracking — larger initial steps and gentler shrinkage
#     # method = Optim.LBFGS(
#     #     linesearch = LineSearches.BackTracking(
#     #       order = 2,      # quadratic interpolation 
#     #       α     = 1e-4,   # Armijo constant (smaller → bolder)
#     #       β     = 0.8     # shrink factor (larger → bolder)
#     #     ),
#     #     m = 20
#     # )

#     # 3) Trust-region (never too timid, never too bold)
#     method = Optim.NewtonTrustRegion()

#     # run
#     res = optimize(safe_f, gradient!, ψ₀, method, opts)

#     if Optim.converged(res)
#         @info "Broyden/BFGS converged in $(res.iterations) steps"
#     else
#         @warn "Broyden/BFGS did NOT converge" status = res
#     end
#     @info "Final objective = $(Optim.minimum(res))"

#     return Optim.minimizer(res)
# end

function newtonOptimizeBroyden(f, ψ₀; tol=1e-6, maxiter=10, verbose=false)
    @info "Starting BFGS optimization" tol=tol maxiter=maxiter

    # — wrap original objective to guard against Infs/NaNs —
    safe_f = x -> begin
        val = try
            f(x)
        catch
            Inf
        end
        return isfinite(val) ? val : typemax(Float64)
    end

    # 1) Build & compile tape on safe objective
    tape = ReverseDiff.GradientTape(safe_f, ψ₀)
    @info "Compiling ReverseDiff gradient tape…"
    ReverseDiff.compile(tape)

    # 2) Mutating gradient! for Optim.jl
    function gradient!(g::AbstractVector, x::AbstractVector)
        ReverseDiff.gradient!(g, tape, x)
        return g
    end

    # 3) Optim options, keep printing behavior
    opts = Optim.Options(
        g_tol      = tol,
        iterations = maxiter,
        show_trace = verbose,
        store_trace= verbose
    )

    # 4) Build LBFGS with backtracking
    method = Optim.LBFGS(linesearch = LineSearches.BackTracking(), m=20)

    # 5) Run optimization on safe objective
    res = optimize(safe_f, gradient!, ψ₀, method, opts)

    # 6) Report
    if Optim.converged(res)
        @info "BFGS converged in $(res.iterations) steps"
    else
        @warn "BFGS did NOT converge" status=res
    end
    @info "Final objective (safe) = $(Optim.minimum(res))"

    return Optim.minimizer(res)
end

""" Shitty ass newton, should not be used.
    newton_krylov(fobj, ψ0;
                  tol=1e-6,
                  maxiter=10,
                  cg_tol=1e-2,
                  cg_maxiter=50)

Newton’s method with a (finite‐difference) Krylov–CG solve of the Hessian system:

1. g = ∇f(ψ) via ReverseDiff
2. Hmv(v) ≈ (∇f(ψ + δ·v) – g)/δ  with δ = √eps()
3. Wrap Hmv in a LinearMap
4. d,stats = cg(Hmap, –g; abstol=cg_tol, reltol=0.0, maxiter=cg_maxiter)
5. ψ ← ψ + d
"""
function newton_krylov(fobj, ψ0; tol=1e-6, maxiter=10, cg_tol=1e-2, cg_maxiter=50)
    ψ = copy(ψ0)
    n = length(ψ)
    δ = sqrt(eps(Float64))

    # ─── Build & compile the tape ONCE ───
    @time grad_tape = ReverseDiff.GradientTape(fobj, ψ0)
    @time ReverseDiff.compile(grad_tape)

    for it in 1:maxiter
        # # 1) gradient via ReverseDiff
        # g = ReverseDiff.gradient(fobj, ψ)
        # ng = norm(g)
        # if ng < tol
        #     println("→ Converged at iter $it with ‖g‖ = $ng")
        #     return ψ
        # end

        # # 2) finite-difference Hessian-vector product
        # Hmv = v -> (ReverseDiff.gradient(fobj, ψ .+ δ .* v) .- g) ./ δ
        # 1) gradient via our precompiled tape
        g = similar(ψ)
        ReverseDiff.gradient!(g, grad_tape, ψ)
        ng = norm(g)
        if ng < tol
            println("→ Converged at iter $it with ‖g‖ = $ng")
            return ψ
        end

        # 2) Hessian-vector using the same tape
        Hmv = v -> begin
            gp = similar(g)
            ReverseDiff.gradient!(gp, grad_tape, ψ .+ δ .* v)
            @. (gp - g) / δ
        end

        # 3) wrap in a LinearMap
        Hmap = LinearMap(Hmv, n, n; issymmetric=true)

        # 4) approximately solve H d = –g
        d, stats = cg(Hmap, -g; abstol=cg_tol, reltol=0.0, maxiter=cg_maxiter)

        its = if stats isa Number
            Int(round(stats))
        elseif hasproperty(stats, :iter)
            stats.iter
        else
            -1
        end
        # 5) update
        ψ .+= d
        @printf("→ Newton %2d: ‖g‖=%.6f   ‖Δψ‖=%.6f   CG its=%d\n",
                it, ng, norm(d), its)
    end

    @warn "→ Hit maxiter without convergence (final ‖g‖ = $(norm(ReverseDiff.gradient(fobj, ψ))))"
    return ψ
end



# (your existing newtonOptimize, newtonOptimizeBroyden go here…)

"""
    newtonOptimizeOptim(fobj, x0; tol, maxiter)

Run Optim.Newton on fobj starting at x0 with forward‐mode Hessian,
printing timing, trace, etc., then return the minimizer.
"""

function newtonOptimizeOptim(fobj, x0; tol=1e-6, maxiter=10)
    # precompute gradient & Hessian so you’re sure they’re working
    @time _ = ReverseDiff.gradient(fobj, x0)
    @time _ = ForwardDiff.hessian(fobj, x0)
    td   = TwiceDifferentiable(fobj, x0; autodiff = :forward, inplace = true)
    opts = Optim.Options(g_tol=tol, iterations=maxiter)
    res  = optimize(td, x0, Newton(), opts)
    return Optim.minimizer(res)
end



end # module
