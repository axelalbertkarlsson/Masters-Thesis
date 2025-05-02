module outputData

using LinearAlgebra

include("pricingFunctions.jl")

using .pricingFunctions

function calculateRateAndRepricing(EAll, zAll, I_z_t, xAll, oAll, oIndAll, tcAll, θg, n_z_t,n_t, n_s, n_u)
    # Initialization
    T=n_t;
    E = EAll[1]
    n_rows = size(θg, 1)  # 3661, consistent with the rest
    fAll = zeros(n_rows, T)    
    priceAll = Vector{Vector{Float64}}(undef, T)
    innovationAll = deepcopy(zAll);

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

        priceAll[t] = g

        innovationAll[t] = zAll[t] - g #Maybe zAll[t] - H_t - u_t
    end

    return fAll, priceAll, innovationAll;
end

end # module
