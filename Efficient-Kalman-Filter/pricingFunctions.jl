module pricingFunctions

# === Exported functions ===
export calcO

# === Functions ===

"""
    calcO(firstDate, tradeDate, theta_g, ecbRatechangeDates, n_c, n_z_t, T0, T)

Calculates the o matrix used in pricing based on extrapolated steps.
"""
function calcO(firstDate, tradeDate, theta_g, ecbRatechangeDates, n_c, n_z_t, T0, T)
    nExtrapolate = Int(tradeDate - firstDate)
    E = [repeat(theta_g[1, :]', nExtrapolate, 1); theta_g]

    datesStep = ecbRatechangeDates[ecbRatechangeDates .> tradeDate]
    datesStep = datesStep[1:Int(n_c)]

    Es = zeros(size(E, 1), Int(n_c))
    for i in 1:Int(n_c)
        Es[Int(datesStep[i] - tradeDate + 1):end, i] .= 1
    end

    E = [E Es]
    intE = [zeros(1, size(E, 2)); cumsum(E, dims=1)] / 365

    o = zeros(0, size(intE, 2))  # initialize 0 rows but right number of columns

    for j in 1:Int(n_z_t)
        col = T[:, j]
        T_nonzero_values = col[col .!= 0.0]
        T_indices = Int.(T_nonzero_values)

        if j == 1
            o = [o; -intE[T_indices, :]]
        else
            row1 = reshape(-intE[Int(T0[j, 1]), :], 1, :)
            o = [o; row1; -intE[T_indices, :]]
        end
    end

    return o
end

end # module
