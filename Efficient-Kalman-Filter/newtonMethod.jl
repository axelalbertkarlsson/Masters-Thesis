# newtonMethodRev.jl
module newtonMethod

using ReverseDiff, LinearAlgebra, Optim, LineSearches, ForwardDiff

export newtonStep, newtonOptimize, newtonOptimizeBroyden

# === Build compiled tapes for gradient & Hessian ===
    function setup_tapes(f::Function, ψ0::AbstractVector{<:AbstractFloat})
        println("→ Building gradient tape…")
        grad_tape = ReverseDiff.GradientTape(f, ψ0)
        println("→ Compiling gradient tape…")
        ReverseDiff.compile(grad_tape)
        println("→ Building Hessian tape…")
        hess_tape = ReverseDiff.HessianTape(f, ψ0)
        println("→ Compiling Hessian tape…")
        ForwardDiff.compile(hess_tape)
        println("→ Done compiling tapes.")
        return grad_tape, hess_tape
      end
      

# === One Newton step, with Armijo backtracking ===
function newtonStep(f, grad_tape, hess_tape,
                    ψ::AbstractVector{<:AbstractFloat};
                    α::Float64=1e-4,
                    s0::Float64=1.0,
                    s_min::Float64=1e-8,
                    verbose::Bool=false)

    n = length(ψ)
    # 1) gradient
    g = zeros(n)
    ReverseDiff.gradient!(g, grad_tape, ψ)

    # 2) Hessian
    H = zeros(n, n)
    ForwardDiff.hessian!(H, hess_tape, ψ)

    # 3) Newton direction Δψ = −H⁻¹ g
    Δψ = - H \ g

    # 4) backtracking line search (Armijo)
    f0 = f(ψ)
    s = s0
    while s ≥ s_min && f(ψ .+ s .* Δψ) > f0 + α * s * dot(g, Δψ)
        s *= 0.5
    end

    if verbose
        println("  ‖Δψ‖ = ", norm(Δψ), "   step size = ", s)
    end

    ψ_new = ψ .+ s .* Δψ
    return ψ_new, norm(Δψ)
end

# === Full Newton optimizer ===
function newtonOptimize(f, ψ₀::AbstractVector{<:AbstractFloat};
                        tol::Float64=1e-4,
                        maxiter::Int=5,
                        verbose::Bool=false)

    ψ = copy(ψ₀)
    grad_tape, hess_tape = setup_tapes(f, ψ₀)

    for k in 1:maxiter
        ψ_new, δnorm = newtonStep(f, grad_tape, hess_tape, ψ; verbose=verbose)
        if δnorm < tol
            verbose && println("Converged at iter = $k (‖Δψ‖=$δnorm)")
            return ψ_new
        end
        ψ = ψ_new
    end

    verbose && println("Reached maxiter = $maxiter (‖Δψ‖=$(δnorm))")
    return ψ
end
function newtonOptimizeBroyden(f, ψ₀; tol=1e-6, maxiter=10, verbose=false)
    @info "Starting BFGS optimization" tol=tol maxiter=maxiter

    # 1) Build & compile tape
    tape = ReverseDiff.GradientTape(f, ψ₀)
    @info "Compiling ReverseDiff gradient tape…"
    ReverseDiff.compile(tape)

    # 2) Mutating gradient! for Optim.jl
    function gradient!(g::AbstractVector, x::AbstractVector)
        # g is pre-allocated by Optim.jl
        ReverseDiff.gradient!(g, tape, x)
        return g
    end

    # 3) Optim options
    # opts = Optim.Options(
    #   g_tol      = tol,
    #   iterations = maxiter,
    #   show_trace = verbose,
    #   store_trace= verbose
    # )
    opts = Optim.Options(g_tol=tol,
                          iterations=maxiter,
                          show_trace=verbose)

    # build a BFGS optimizer that uses plain backtracking
    bt = LineSearches.BackTracking()
    method = Optim.BFGS(linesearch = bt)

    res = optimize(f, gradient!, ψ₀, method, opts)

    # 4) Run BFGS (opts must be positional arg #5)
    # res = optimize(f, gradient!, ψ₀, BFGS(), opts)

    # 5) Report
    if Optim.converged(res)
        @info "BFGS converged in $(res.iterations) steps"
    else
        @warn "BFGS did NOT converge" status=res
    end
    @info "Final objective" f_min=Optim.minimum(res)

    return Optim.minimizer(res)
end

end # module
