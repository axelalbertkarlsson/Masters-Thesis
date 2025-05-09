module pricingFunctions

# === Exported functions ===
export calcO, taylorApprox

# === Functions ===

"""
    calcO(firstDate, tradeDate, theta_g, ecbRatechangeDates, n_c, n_z_t, T0, T)

Calculates the o matrix used in pricing based on extrapolated steps.
"""
function calcO(firstDate, tradeDate, theta_g, ecbRatechangeDates, n_c, n_z_t, T0, T)
    nExtrapolate = Int(tradeDate) - Int(firstDate)

    # vertical stack via cat instead of [ ... ; ... ]
    E = cat(
      repeat(theta_g[1, :]', nExtrapolate, 1),
      theta_g;
      dims=1
    )

    n_steps = 16
    datesStep = ecbRatechangeDates[ecbRatechangeDates .> tradeDate]
    datesStep = datesStep[1:Int(n_steps)]

    Es = zeros(size(E, 1), Int(n_steps))
    for i in 1:Int(n_steps)
        Es[(Int(datesStep[i]) - Int(tradeDate) + 1):end, i] .= 1
    end

    # horizontal concat is safe, you can keep [E Es] or do:
    E = cat(E, Es; dims=2)

    # again vertical stack via cat
    intE = cat(
      zeros(1, size(E, 2)),
      cumsum(E, dims=1);
      dims=1
    ) / 365

    o = zeros(0, size(intE, 2))

    for j in 1:Int(n_z_t)
        col = T[:, j]
        T_nonzero_values = col[col .!= 0.0]
        T_indices = Int.(T_nonzero_values)

        if j == 1
            o = cat(o, -intE[T_indices, :]; dims=1)
        else
            row1 = reshape(-intE[Int(T0[j, 1]), :], 1, :)
            o = cat(o, row1; dims=1)
            o = cat(o, -intE[T_indices, :]; dims=1)
        end
    end

    return o, E
end

"""
taylorApprox(o, oInd, tc, I_z, n_c)

Computes the first-order Taylor approximation (g, G) for a set of instruments.

"""
function taylorApprox(o, oInd, tc, x, I_z, n_z_t)
    n_z_t = Int(n_z_t)
    oInd = Int.(oInd)

    g = Vector{eltype(x)}(undef, n_z_t)
    G = Matrix{eltype(x)}(undef, n_z_t, size(x, 1))

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
            if any(isnan, G[j, :])
                println("G[$j, :] contains NaN at j = $j, ind = $ind")
            end
        end
    end
    H = [G I_z]
    u = g - G*x
    return H, u, g, G
end


end # module
