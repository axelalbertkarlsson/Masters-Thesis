module outputData

using LinearAlgebra

function calculateRateAndRepricing(EAll, zAll, I_z_t, xAll, oAll, oIndAll, tcAll, θg, n_z_t,n_t, n_s)
    # Initialization
    T=n_t;
    E = EAll[1]
    n_rows = size(θg, 1)  # 3661, consistent with the rest
    fAll = zeros(n_rows, T)    
    priceAll = Vector{Vector{Float64}}(undef, T)

    for t in 1:T

        z = zAll[t]
        I_z = I_z_t[t]
        x = xAll[t]

        x_s = x[1:end - n_s]

        E = EAll[t];

        E = E[end - 3660:end, :]  # Keep only the last 3661 rows, otherwise mismatch

        fAll[:, t] = [θg E] * x_s

        # Repricing errors (model-implied prices)
        price = Float64[]
        # o_inst = oAll[t]
        # dTtmp = year_fracAll[t]

        # for j in 1:n_z_t[t]
        #     o_tmp = o_inst[j]
        #     deltaT = dTtmp[j]

        #     o_x = o_tmp * x_s
        #     price = (exp(o_x[1]) - exp(o_x[end])) / (exp.(o_x[2:end])' * deltaT)
        #     push!(g, price)
        # end

        priceAll[t] = price
    end

    return fAll, priceAll
end

end # module
