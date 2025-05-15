module outputData


include("pricingFunctions.jl")

using LinearAlgebra, Printf
using .pricingFunctions

import Statistics

export calculateMSE, calculateRateAndRepricing

function calculateRateAndRepricing(EAll, zAll, I_z_t, xAll, oAll, oIndAll, tcAll, θg, n_z_t,n_t, n_s, n_u)
    # Initialization
    T=n_t;
    E = EAll[1]
    n_rows = size(θg, 1)  # 3661, consistent with the rest
    fAll = zeros(n_rows, T)    
    zPredAll = Vector{Vector{Float64}}(undef, T)
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

        zPredAll[t] = H_t*x + u_t

        innovationAll[t] = zAll[t] - H_t*x - u_t #Maybe zAll[t] - H_t*x_pred -u_t
    end

    return fAll, zPredAll, innovationAll;
end


function calculateMSE(innovationAll)
    # === Print overall error metrics for both methods ===
    # 1) flatten into one long numeric vector
    all_reg = vcat([vec(x) for x in innovationAll]...)

    # 2a) compute MSE
    mse_reg = Statistics.mean(all_reg .^ 2)

    # 2b) compute MAE
    mae_reg = Statistics.mean(abs.(all_reg))

    # 3) pretty-print with Printf
    println("--------------------------------------------------")
    @printf("MSE  (EKF):           % .5e\n", mse_reg)
    println()
    @printf("MAE  (EKF):           % .5e\n", mae_reg)
    println("--------------------------------------------------")

    return mse_reg, mae_reg    
end

end # module
