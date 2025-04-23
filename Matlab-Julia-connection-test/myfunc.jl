module MyJuliaModule

using Zygote

export square_number, compute_gradients_and_loss

function square_number(x::Float64)
    return x^2
end

function compute_gradients_and_loss(input_matrix::Matrix{Float64})
    f(v) = v[1]^2 + 3 * (v[1] + v[2]) + 5 * v[3]

    n_rows, n_cols = size(input_matrix)
    gradients = zeros(n_rows, n_cols)
    losses = zeros(n_rows)

    for i in 1:n_rows
        x = vec(input_matrix[i, :])
        losses[i] = f(x)
        gradients[i, :] = Zygote.gradient(f, x)[1]
    end

    return (losses, gradients)
end

end # module
