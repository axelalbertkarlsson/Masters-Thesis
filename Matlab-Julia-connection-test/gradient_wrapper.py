import numpy as np
from julia import Julia
jl = Julia(compiled_modules=False)

from julia import Main
Main.include("julia/myfunc.jl")
jlmod = Main.MyJuliaModule

def square_from_julia(x):
    return jlmod.square_number(float(x))

def compute_gradients_and_loss_py(input_matrix):
    matrix = np.array(input_matrix, dtype=np.float64).tolist()
    losses, grads = jlmod.compute_gradients_and_loss(matrix)
    return {
        "losses": list(losses),
        "gradients": [list(g) for g in grads]
    }

# # Example test run
# if __name__ == "__main__":
#     mat = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
#     result = compute_gradients_and_loss_py(mat)
#     print("Losses:", result["losses"])
#     print("Gradients:", result["gradients"])