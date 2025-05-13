module newtonMethod

using ReverseDiff, LinearAlgebra
using Optim, LineSearches
using ForwardDiff

export newtonStep, newtonOptimize, newtonOptimizeBroyden, optimize_parameters, optimize_bfgs

# === Build & compile a single ReverseDiff tape for gradients ===
function setup_tape!(f::Function, x0::Vector{Float64})
    @info "Building gradient tape…"
    tape = ReverseDiff.GradientTape(f, x0)
    @info "Compiling gradient tape…"
    ReverseDiff.compile(tape)
    return tape
end

# === One Newton step, with Armijo backtracking ===
function newtonStep(f::Function, tape, x::Vector{Float64};
                    α::Float64=1e-4, s0::Float64=1.0, s_min::Float64=1e-8,
                    verbose::Bool=false)
    n = length(x)
    g = zeros(n)
    ReverseDiff.gradient!(g, tape, x)
    H = ForwardDiff.hessian!(zeros(n,n), y->f(y), x)
    Δ = -H \ g

    f0 = f(x); s = s0
    while s ≥ s_min && f(x .+ s .* Δ) > f0 + α * s * dot(g, Δ)
        s *= 0.5
    end
    verbose && @info " ‖Δ‖=$(norm(Δ)), step=$s"

    return x .+ s .* Δ, norm(Δ)
end

# === Full Newton optimizer ===
function newtonOptimize(f::Function, x0::Vector{Float64};
                        tol::Float64=1e-4, maxiter::Int=5, verbose::Bool=false)
    tape = setup_tape!(f, x0)
    x = copy(x0)
    for k in 1:maxiter
        x_new, δ = newtonStep(f, tape, x; verbose=verbose)
        if δ < tol
            verbose && @info "Converged at iter=$k (‖Δ‖=$δ)"
            return x_new
        end
        x = x_new
    end
    verbose && @warn "Reached maxiter=$maxiter (‖Δ‖=$(δ))"
    return x
end

# === Unconstrained BFGS via ReverseDiff + Optim.jl ===
function newtonOptimizeBroyden(f::Function, x0::Vector{Float64};
                              tol::Float64=1e-6, maxiter::Int=10, verbose::Bool=false)
    @info "Starting unconstrained BFGS…"
    tape = setup_tape!(f, x0)
    grad! = (g,x)->(ReverseDiff.gradient!(g,tape,x); g)

    opts = Optim.Options(
      g_tol      = tol,
      iterations = maxiter,
      store_trace= verbose,
      show_trace = verbose
    )
    res = optimize(f, grad!, x0, BFGS(), opts)
    Optim.converged(res) ?
      @info("BFGS converged in $(res.iterations) steps") :
      @warn("BFGS did NOT converge", status=res)
    return Optim.minimizer(res)
end

"""
 One‐stop wrapper for quasi‐Newton (BFGS) *without* box‐constraints:

 - `f`       : objective f(x::Vector) → Float64  
 - `x0`      : initial guess (Vector{Float64})  
 - `tol`     : gradient‐norm tolerance  
 - `maxiter` : max iterations  
 - `verbose` : trace output  

Returns the minimizer as Vector{Float64}.
"""
function optimize_parameters(
    f::Function,
    x0::Vector{Float64};
    tol::Float64      = 1e-6,
    maxiter::Int      = 10,
    verbose::Bool     = false
)
    # compile tape + gradient mutator
    tape = setup_tape!(f, x0)
    grad! = (g,x)->(ReverseDiff.gradient!(g,tape,x); g)

    # build a BFGS instance with a modest initial step and backtracking
    inner_method = BFGS(
      alphaguess = InitialStatic(alpha=1e-2),
      linesearch = BackTracking()
    )

    # inner‐solver options
    opts = Optim.Options(
      g_tol      = tol,
      iterations = maxiter,
      store_trace= verbose,
      show_trace = verbose
    )

    @info "Starting unconstrained BFGS…"
    res = optimize(f, grad!, x0, inner_method, opts)
    return Optim.minimizer(res)
end

function optimize_bfgs(f, x0; tol=1e-6, maxiter=10, verbose=false)
    res = optimize(
      f,                             # objective
      x0,                            # initial guess
      BFGS();                        # use BFGS
      autodiff   = :forward,         # ForwardDiff for gradients
      g_tol      = tol,
      iterations = maxiter,
      show_trace = verbose,
      store_trace= verbose
    )
    return Optim.minimizer(res)
end

end # module
