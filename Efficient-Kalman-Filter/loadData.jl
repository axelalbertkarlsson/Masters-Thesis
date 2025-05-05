module loadData

using MAT

# === Exported items ===
export KalmanData, load_all_data, run, split_data, convert_to_f32

# === Internal Structs ===

struct ObservedData
    ecbRatechangeDates   # Vector{Float64}
    zAll                 # Matrix{Float64}
    times                # Vector{Float64}
end

struct PricingData
    firstDates           # Vector{Float64}
    idContracts          # Vector{Any}
    TAll                 # Vector{Matrix{Float64}}
    T0All                # Vector{Matrix{Float64}}
    oIndAll              # Vector{Any}
    tcAll                # Vector{Any}
    tradeDates           # Vector{Float64}
end

struct Psi0Data
    Sigma_v              # Matrix{Float64}
    Sigma_w              # Matrix{Float64}
    Sigma_x              # Matrix{Float64}
    a_x                  # Vector{Float64}
    theta_F              # Vector{Float64}
    theta_g              # Matrix{Float64}
end

struct RefKFVariables
    A_t                  # Vector{Matrix{Float64}}
    B_t                  # Vector{Matrix{Float64}}
    D_t                  # Vector{Matrix{Float64}}
    G_t                  # Vector{Matrix{Float64}}
    I_z_t                # Vector{Matrix{Float64}}
    f_t                  # Matrix{Float64}
    n_c                  # Int
    n_p                  # Int
    n_s                  # Int
    n_t                  # Int
    n_u                  # Int
    n_x                  # Int
    n_z_t                # Vector{Float64}
end

struct KalmanData
    ecbRatechangeDates
    zAll
    times

    firstDates
    idContracts
    TAll
    T0All
    oIndAll
    tcAll
    tradeDates

    Sigma_v
    Sigma_w
    Sigma_x
    a_x
    theta_F
    theta_g

    A_t
    B_t
    D_t
    G_t
    I_z_t

    f_t

    n_c
    n_p
    n_s
    n_t
    n_u
    n_x
    n_z_t
end

# === Function to load all .mat files (Float64) ===

function load_all_data(data_folder::String)
    vars = Dict{Symbol, Any}()
    files = ["observedData.mat", "pricingData.mat", "psi_0.mat", "refKFVariables.mat"]

    for file in files
        matopen(joinpath(data_folder, file)) do f
            for varname in names(f)
                vars[Symbol(varname)] = read(f, varname)
            end
        end
    end

    obs = ObservedData(vars[:ecbRatechangeDates], vars[:zAll], vars[:times])
    prc = PricingData(vars[:firstDates], vars[:idContracts], vars[:TAll], vars[:T0All], vars[:oIndAll], vars[:tcAll], vars[:tradeDates])
    psi = Psi0Data(vars[:Sigma_v], vars[:Sigma_w], vars[:Sigma_x], vars[:a_x], vars[:theta_F], vars[:theta_g])
    refkf = RefKFVariables(vars[:A_t], vars[:B_t], vars[:D_t], vars[:G_t], vars[:I_z_t], vars[:f_t], vars[:n_c], vars[:n_p], vars[:n_s], vars[:n_t], vars[:n_u], vars[:n_x], vars[:n_z_t])

    return (obs, prc, psi, refkf)
end

# === Run loader and return KalmanData ===

function run(data_folder::String = "Efficient-Kalman-Filter/Data")
    obs, prc, psi, refkf = load_all_data(data_folder)
    return KalmanData(
        obs.ecbRatechangeDates, obs.zAll, obs.times,
        prc.firstDates, prc.idContracts, prc.TAll, prc.T0All, prc.oIndAll, prc.tcAll, prc.tradeDates,
        psi.Sigma_v, psi.Sigma_w, psi.Sigma_x, psi.a_x, psi.theta_F, psi.theta_g,
        refkf.A_t, refkf.B_t, refkf.D_t, refkf.G_t, refkf.I_z_t, refkf.f_t,
        refkf.n_c, refkf.n_p, refkf.n_s, refkf.n_t, refkf.n_u, refkf.n_x, refkf.n_z_t
    )
end

# === Split data into in-sample and out-of-sample ===

function split_data(data::KalmanData, ratio::Float64)
    # Use actual number of time steps from zAll rows
    n_total = size(data.zAll, 1)
    split_idx = Int(floor(ratio * n_total))
    ins = run_split(data, 1, split_idx)
    out = run_split(data, split_idx+1, n_total)
    return (insample = ins, outsample = out)
end

# Helper to slice KalmanData between row indices i1 and i2
function run_split(data::KalmanData, i1::Int, i2::Int)
    return KalmanData(
        data.ecbRatechangeDates,
        data.zAll[i1:i2, :],
        data.times[i1:i2],
        data.firstDates,
        data.idContracts,
        data.TAll[i1:i2],
        data.T0All[i1:i2],
        data.oIndAll[i1:i2],
        data.tcAll[i1:i2],
        data.tradeDates[i1:i2],
        data.Sigma_v,
        data.Sigma_w,
        data.Sigma_x,
        data.a_x,
        data.theta_F,
        data.theta_g,
        data.A_t[i1:i2],
        data.B_t[i1:i2],
        data.D_t[i1:i2],
        data.G_t[i1:i2],
        data.I_z_t[i1:i2],
        data.f_t[i1:i2, :],
        data.n_c,
        data.n_p,
        data.n_s,
        i2 - i1 + 1,
        data.n_u,
        data.n_x,
        data.n_z_t[i1:i2]
    )
end

end # module

function run_split(data, i1, i2)
    return KalmanData(
        data.ecbRatechangeDates,
        data.zAll[i1:i2, :],
        data.times[i1:i2],
        data.firstDates,
        data.idContracts,
        data.TAll[i1:i2],
        data.T0All[i1:i2],
        data.oIndAll[i1:i2],
        data.tcAll[i1:i2],
        data.tradeDates[i1:i2],
        data.Sigma_v,
        data.Sigma_w,
        data.Sigma_x,
        data.a_x,
        data.theta_F,
        data.theta_g,
        data.A_t[i1:i2],
        data.B_t[i1:i2],
        data.D_t[i1:i2],
        data.G_t[i1:i2],
        data.I_z_t[i1:i2],
        data.f_t[i1:i2, :],
        data.n_c,
        data.n_p,
        data.n_s,
        i2 - i1 + 1,
        data.n_u,
        data.n_x,
        data.n_z_t[i1:i2]
    )
end

# === Post-conversion to Float32 ===
"""
convert_to_f32(data::KalmanData)

Convert every Float64 array in KalmanData to Float32, and cast count fields to Int.
Non-numeric fields remain unchanged.
"""
function convert_to_f32(d::KalmanData)
    return KalmanData(
        Float32.(d.ecbRatechangeDates),
        Float32.(d.zAll),
        Float32.(d.times),

        Float32.(d.firstDates),
        d.idContracts,
        [Float32.(m) for m in d.TAll],
        [Float32.(m) for m in d.T0All],
        d.oIndAll,
        d.tcAll,
        Float32.(d.tradeDates),

        Float32.(d.Sigma_v),
        Float32.(d.Sigma_w),
        Float32.(d.Sigma_x),
        Float32.(d.a_x),
        Float32.(d.theta_F),
        Float32.(d.theta_g),

        [Float32.(m) for m in d.A_t],
        [Float32.(m) for m in d.B_t],
        [Float32.(m) for m in d.D_t],
        [Float32.(m) for m in d.G_t],
        [Float32.(m) for m in d.I_z_t],

        Float32.(d.f_t),

        Int(d.n_c),
        Int(d.n_p),
        Int(d.n_s),
        Int(d.n_t),
        Int(d.n_u),
        Int(d.n_x),
        Float32.(d.n_z_t)
    )
end


