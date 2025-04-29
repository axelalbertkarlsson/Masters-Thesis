module pricingFunctions

# === Exported functions ===
export calcO, taylorApprox

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

"""
taylorApprox(o, oInd, tc, I_z, n_c)

Computes the first-order Taylor approximation (g, G) for a set of instruments.

"""
function taylorApprox(o, oInd, tc, x, I_z, n_z_t)
    n_z_t = Int(n_z_t)
    oInd = Int.(oInd)

    g = zeros(n_z_t)
    G = zeros(n_z_t, size(x, 1))

    for j in 1:n_z_t
        ind = oInd[j]:(oInd[j+1]-1)
        if j == 1
            eox = exp.(-o[ind, :] * x)
            g[j] = (eox[1] - 1) / tc[ind][1]
            G[j, :] = (-eox[1]) .* o[ind, :] ./ tc[ind][1]

        else
            eox = exp.(o[ind, :] * x)
            den = sum(tc[ind[2:end]] .* eox[2:end])
            g[j] = (eox[1] - eox[end]) / den
            G[j, :] = (eox[1] .* o[ind[1], :] .- eox[end] .* o[ind[end], :]) ./ den
            #sum_part = sum(tc[ind[2:end]] .* eox[2:end])
            weighted_sum = sum((tc[ind[2:end]] .* eox[2:end]) .* o[ind[2:end], :], dims=1)
            G[j, :] .-= ((eox[1] - eox[end]) / den^2) .* vec(weighted_sum)
        end
    end
    H = [G I_z]
    u = g - G*x
    return H, u, g, G
end


end # module
