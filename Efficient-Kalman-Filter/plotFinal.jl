using MAT, Revise

include("loadData.jl")
include("outputData.jl")
include("EKF.jl")

using .loadData, .EKF,..outputData

function load_mat_vars(file_path::String)
    vars = Dict{Symbol,Any}()
    matopen(file_path) do f
        for name in names(f)
            vars[Symbol(name)] = read(f, name)
        end
    end
    return vars
end

# Utility to load all variables from a `.mat` file into a Dict{Symbol,Any}
function load_psi_tuple(filename::String)
    vars = load_mat_vars(filename)
    return (
      # exactly the same order & shapes as ψ₀ above:
      Float64.(vars[:Sigma_w]),        # 1 ⇒ Σ𝓌
      Float64.(vars[:Sigma_v]),        # 2 ⇒ Σᵥ
      Float64.(vec(vars[:a0])),        # 3 ⇒ a₀
      Float64.(vars[:Sigma_x]),        # 4 ⇒ Σₓ
      Float64.(vec(vars[:theta_F])),   # 5 ⇒ θ𝐹
      Float64.(vars[:theta_g])         # 6 ⇒ θ𝗀
    )
end

function getResults(ψ, outs)
    Σw, Σv, a0, Σx, θF, θg = ψ
    x_f, P_f, x_s, P_s, P_l, oAll, EAll =
    EKF.kalman_filter_smoother_lag1(
      outs.zAll, outs.oIndAll, outs.tcAll, outs.I_z_t, outs.f_t,
      outs.n_c, outs.n_p, outs.n_s, outs.n_t,
      outs.n_u, outs.n_x, Int.(outs.n_z_t),
      outs.A_t, outs.B_t, outs.D_t, outs.G_t,
      Σw, Σv, a0, Σx, θF, θg,
      outs.firstDates, outs.tradeDates,
      outs.ecbRatechangeDates, outs.T0All, outs.TAll
    )
    fAll, _, innov, _ = outputData.calculateRateAndRepricing(
    EAll, outs.zAll, outs.I_z_t, x_s, oAll,
    outs.oIndAll, outs.tcAll, θg,
    Int.(outs.n_z_t), outs.n_t, outs.n_s, outs.n_u, outs.G_t, Σv, P_f  
  )
  return fAll, innov
end

function write_results(
    filename::AbstractString,
    fAll, innovationAll, times
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
        write(f, "fAll", fAll)
        write(f, "innovationAll",         cellify(innovationAll))
        write(f, "times",                    cellify(times))
    end
end

println("Loading initial data...")
data = loadData.run(joinpath("Efficient-Kalman-Filter","Data"))
# split = loadData.split_data(data, 0.5)
# ins, outs = split.insample, split.outsample

# Load psi_final_NM variables
println("Loading psi_final_NM.mat...")
ψ_NM = load_psi_tuple("psi_final_NM.mat")

# Load psi_final_EM variables
println("Loading psi_final_EM.mat...")
ψ_EM = load_psi_tuple("psi_final_EM.mat")

ψ_0 = (
  data.Sigma_w,
  data.Sigma_v,
  vec(data.a_x),
  data.Sigma_x,
  vec(data.theta_F),
  data.theta_g
)

println("Getting results RKF...")
fAll_RKF, innov_RKF = getResults(ψ_0, data)
println("Writing results RKF...")
write_results("final_Reg.mat", fAll_RKF, innov_RKF, data.times)

println("Getting results NM...")
fAll_NM, innov_NM = getResults(ψ_NM, data)
println("Writing results NM...")
write_results("final_NM.mat", fAll_NM, innov_NM, data.times)

println("Getting results EM...")
fAll_EM, innov_EM = getResults(ψ_EM, data)
println("Writing results EM...")
write_results("final_EM.mat", fAll_EM, innov_EM, data.times)



