# newtonMethodRev.jl
module newtonMethod

using ReverseDiff, LinearAlgebra

export newtonStep, newtonOptimize

# === Build compiled tapes for gradient & Hessian ===
function setup_tapes(f::Function, ψ0::AbstractVector{<:Float64})
    grad_tape = ReverseDiff.GradientTape(f, ψ0)
    ReverseDiff.compile(grad_tape)
    hess_tape = ReverseDiff.HessianTape(f, ψ0)
    ReverseDiff.compile(hess_tape)
    return grad_tape, hess_tape
end

# === One Newton step, with Armijo backtracking ===
function newtonStep(f, grad_tape, hess_tape,
                    ψ::AbstractVector{<:Float64};
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
    ReverseDiff.hessian!(H, hess_tape, ψ)

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
function newtonOptimize(f, ψ₀::AbstractVector{<:Float64};
                        tol::Float64=1e-6,
                        maxiter::Int=50,
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

end # module
