module outputData


include("pricingFunctions.jl")

using LinearAlgebra, Printf, MAT
using .pricingFunctions

import Statistics

export calculateMSE, calculateRateAndRepricing, write_results

function calculateRateAndRepricing(EAll, zAll, I_z_t, xAll, oAll, oIndAll, tcAll, θg, n_z_t,n_t, n_s, n_u,  GAll, Σv, P_predAll)
    # Initialization
    T=n_t;
    E = EAll[1]
    n_rows = size(θg, 1)  # 3661, consistent with the rest
    fAll = zeros(n_rows, T)    
    zPredAll = Vector{Vector{Float64}}(undef, T)
    innovationAll = deepcopy(zAll);
    innovationLik = 0.0

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

        # 2) build the innovation covariance
        S = H_t * P_predAll[t] * H_t' +
        GAll[t] * Σv * GAll[t]'

       # 3) accumulate the log‐likelihood
        innovationLik -= 0.5 * (
            n_z_t[t]*log(2π) +        # dimension term
            logdet(S) +               # log-determinant
            innovationAll[t]' * (S \ innovationAll[t])              # Mahalanobis term
        )

    end

    return fAll, zPredAll, innovationAll, innovationLik;
end


function calculateMSE(innovationAll)
    # === Print overall error metrics for both methods ===
    # 1) flatten into one long numeric vector
    all_reg = vcat([vec(x) for x in innovationAll]...)

    # 2a) compute MSE
    mse_reg = Statistics.mean(all_reg .^ 2)

    # 2b) compute MAE
    mae_reg = Statistics.mean(abs.(all_reg))

    return mse_reg, mae_reg    
end

function write_results(
    filename::AbstractString,
    fAll_NM, zPredNMAll, innovationAll_NM, innovation_likelihood_NM, times_NM, alloc_NM, iters_NM,
    fAll_EM, zPredEMAll, innovationAll_EM, innovation_likelihood_EM, times_EM, alloc_EM, iters_EM,
    zPredRKFAll, innovationAll_RKF,
    times
)
    # helper: wrap every element (scalar or vector) into a column vector
    function cellify(x)
        cells = Vector{Any}(undef, length(x))
        for (i,v) in enumerate(x)
            arr = v isa AbstractArray ? v : [v]    # wrap scalars
            cells[i] = reshape(arr, :, 1)          # make it a column
        end
        return cells
    end

    matopen(filename, "w") do f
        write(f, "fAll_NM", fAll_NM)
        write(f, "fAll_EM", fAll_EM)

        write(f, "zPredNMAll",               cellify(zPredNMAll))
        write(f, "innovationAll_NM",         cellify(innovationAll_NM))
        write(f, "innovation_likelihood_NM", cellify(innovation_likelihood_NM))
        write(f, "times_NM", times_NM)
        write(f, "alloc_NM", alloc_NM)
        write(f, "iters_NM", iters_NM)

        write(f, "zPredEMAll",               cellify(zPredEMAll))
        write(f, "innovationAll_EM",         cellify(innovationAll_EM))
        write(f, "innovation_likelihood_EM", cellify(innovation_likelihood_EM))
        write(f, "times_EM", times_EM)
        write(f, "alloc_EM", alloc_EM)
        write(f, "iters_EM", iters_EM)

        write(f, "zPredRKFAll",              cellify(zPredRKFAll))
        write(f, "innovationAll_RKF",        cellify(innovationAll_RKF))

        write(f, "times",                    cellify(times))
    end
end

end # module
