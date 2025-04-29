module loadData

using MAT

# === Exported items ===
export KalmanData, load_all_data, run

# === Internal Structs ===

struct ObservedData
    ecbRatechangeDates   # Vector{Float64} - ECB rate change dates
    zAll                 # Matrix{Float64} - Observed data over time
end

struct PricingData
    firstDates           # Vector{Float64} - Contract start dates
    idContracts          # Vector{Any}     - Contract IDs (cell array)
    TAll                 # Vector{Matrix}  - New: TAll matrices (5030x1, each 10x28)
    T0All                # Vector{Matrix}  - New: T0All vectors (5030x1, each 28x1)
    oIndAll              # Vector{Any}     - Original indices
    tcAll                # Vector{Any}     - Transaction costs/types
    tradeDates           # Vector{Float64} - Trade dates
end

struct Psi0Data
    Sigma_v              # Matrix{Float64} (28x28) - Measurement noise covariance
    Sigma_w              # Matrix{Float64} (50x50) - Process noise covariance
    Sigma_x              # Matrix{Float64} (50x50) - Initial state covariance
    a_x                  # Vector{Float64} (50x1)  - Initial state mean
    theta_F              # Vector{Float64} (50x1)  - F matrix parameters
    theta_g              # Matrix{Float64} (3661x6) - g matrix parameters
end

struct RefKFVariables
    A_t                  # Vector{Matrix} - Transition matrices A_t[time][i,j]
    B_t                  # Vector{Matrix} - System matrices B_t[time][i,j]
    D_t                  # Vector{Matrix} - System matrices D_t[time][i,j]
    G_t                  # Vector{Matrix} - Measurement matrices G_t[time][i,j]
    I_z_t                # Vector{Matrix} - Mapping matrices I_z_t[time][i,j]
    f_t                  # Matrix{Float64} (5030x3670) - Factors over time
    n_c                  # Int - Number of country factors
    n_p                  # Int - Number of pricing factors
    n_s                  # Int - Number of state shocks
    n_t                  # Int - Number of time steps
    n_u                  # Int - Number of measurement shocks
    n_x                  # Int - State vector dimension
    n_z_t                # Vector{Float64} - Number of observed variables at each time
end

# === Final Struct with all variables grouped ===

"""
    KalmanData

Holds all loaded data from the Efficient Kalman Filter project.

Fields:
- `ecbRatechangeDates::Vector{Float64}` — ECB rate change dates
- `zAll::Matrix{Float64}` — Observed data over time
- `firstDates::Vector{Float64}` — Contract start dates
- `idContracts::Vector{Any}` — Contract IDs
- `TAll::Vector{Matrix}` — T matrices (each 10x28)
- `T0All::Vector{Matrix}` — T0 vectors (each 28x1)
- `oIndAll::Vector{Any}` — Original indices
- `tcAll::Vector{Any}` — Transaction costs/types
- `tradeDates::Vector{Float64}` — Trade dates
- `Sigma_v::Matrix{Float64}` — Measurement noise covariance (28x28)
- `Sigma_w::Matrix{Float64}` — Process noise covariance (50x50)
- `Sigma_x::Matrix{Float64}` — Initial state covariance (50x50)
- `a_x::Vector{Float64}` — Initial state mean (50x1)
- `theta_F::Vector{Float64}` — F matrix parameters (50x1)
- `theta_g::Matrix{Float64}` — g matrix parameters (3661x6)
- `A_t::Vector{Matrix}` — Transition matrices over time
- `B_t::Vector{Matrix}` — System matrices over time
- `D_t::Vector{Matrix}` — System matrices over time
- `G_t::Vector{Matrix}` — Measurement matrices over time
- `I_z_t::Vector{Matrix}` — Mapping matrices over time
- `f_t::Matrix{Float64}` — Factors over time (5030x3670)
- `n_c::Int` — Number of country factors
- `n_p::Int` — Number of pricing factors
- `n_s::Int` — Number of state shocks
- `n_t::Int` — Number of time steps
- `n_u::Int` — Number of measurement shocks
- `n_x::Int` — State vector dimension
- `n_z_t::Vector{Float64}` — Number of observed variables at each time
"""
struct KalmanData
    ecbRatechangeDates
    zAll

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

# === Function to load all .mat files ===

function load_all_data(data_folder::String)
    vars = Dict{Symbol, Any}()
    files = [
        "observedData.mat",
        "pricingData.mat",
        "psi_0.mat",
        "refKFVariables.mat"
    ]
    for file in files
        path = joinpath(data_folder, file)
        matopen(path) do f
            for varname in names(f)
                vars[Symbol(varname)] = read(f, varname)
            end
        end
    end
    return (
        ObservedData(vars[:ecbRatechangeDates], vars[:zAll]),
        PricingData(vars[:firstDates], vars[:idContracts], vars[:TAll], vars[:T0All], vars[:oIndAll], vars[:tcAll], vars[:tradeDates]),
        Psi0Data(vars[:Sigma_v], vars[:Sigma_w], vars[:Sigma_x], vars[:a_x], vars[:theta_F], vars[:theta_g]),
        RefKFVariables(vars[:A_t], vars[:B_t], vars[:D_t], vars[:G_t], vars[:I_z_t], vars[:f_t],
                       vars[:n_c], vars[:n_p], vars[:n_s], vars[:n_t], vars[:n_u], vars[:n_x], vars[:n_z_t])
    )
end

# === Final function: returns KalmanData ready to use ===

function run(data_folder::String = "Efficient-Kalman-Filter/Data")
    observedData, pricingData, psi0Data, refkfData = load_all_data(data_folder)

    return KalmanData(
        observedData.ecbRatechangeDates,
        observedData.zAll,

        pricingData.firstDates,
        pricingData.idContracts,
        pricingData.TAll,
        pricingData.T0All,
        pricingData.oIndAll,
        pricingData.tcAll,
        pricingData.tradeDates,

        psi0Data.Sigma_v,
        psi0Data.Sigma_w,
        psi0Data.Sigma_x,
        psi0Data.a_x,
        psi0Data.theta_F,
        psi0Data.theta_g,

        refkfData.A_t,
        refkfData.B_t,
        refkfData.D_t,
        refkfData.G_t,
        refkfData.I_z_t,

        refkfData.f_t,

        refkfData.n_c,
        refkfData.n_p,
        refkfData.n_s,
        refkfData.n_t,
        refkfData.n_u,
        refkfData.n_x,
        refkfData.n_z_t
    )
end

end # module
