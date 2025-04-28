module loadData

using MAT

# === Exported items ===
export KalmanData, load_all_data, run

# === Internal Structs ===

struct ObservedData
    ecbRatechangeDates
    zAll
end

struct PricingData
    firstDates
    idContracts
    infoInstrAll
    oIndAll
    tcAll
    tradeDates
end

struct Psi0Data
    Sigma_v
    Sigma_w
    Sigma_x
    a_x
    theta_F
    theta_g
end

struct RefKFVariables
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

# === Final Struct with all variables grouped ===

struct KalmanData
    ecbRatechangeDates
    zAll
    firstDates
    idContracts
    infoInstrAll
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
        PricingData(vars[:firstDates], vars[:idContracts], vars[:infoInstrAll], vars[:oIndAll], vars[:tcAll], vars[:tradeDates]),
        Psi0Data(vars[:Sigma_v], vars[:Sigma_w], vars[:Sigma_x], vars[:a_x], vars[:theta_F], vars[:theta_g]),
        RefKFVariables(vars[:A_t], vars[:B_t], vars[:D_t], vars[:G_t], vars[:I_z_t], vars[:f_t],
                       vars[:n_c], vars[:n_p], vars[:n_s], vars[:n_t], vars[:n_u], vars[:n_x], vars[:n_z_t])
    )
end

# === Final function: returns KalmanData ===

function run(data_folder::String = "Efficient-Kalman-Filter/Data")
    observedData, pricingData, psi0Data, refkfData = load_all_data(data_folder)

    return KalmanData(
        observedData.ecbRatechangeDates,
        observedData.zAll,
        pricingData.firstDates,
        pricingData.idContracts,
        pricingData.infoInstrAll,
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
