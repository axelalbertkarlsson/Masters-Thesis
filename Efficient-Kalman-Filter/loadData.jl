module loadData

using MAT

export KalmanData, run, split_data

# === Parameterized data types ===
struct ObservedData{T}
    ecbRatechangeDates::Vector{T}
    zAll::Matrix{T}
    times::Vector{T}
end

struct PricingData{T}
    firstDates::Vector{T}
    idContracts::Vector{Any}
    TAll::Vector{Matrix{T}}
    T0All::Vector{Matrix{T}}
    oIndAll::Vector{Any}
    tcAll::Vector{Any}
    tradeDates::Vector{T}
end

struct Psi0Data{T}
    Sigma_v::Matrix{T}
    Sigma_w::Matrix{T}
    Sigma_x::Matrix{T}
    a_x::Vector{T}
    theta_F::Vector{T}
    theta_g::Matrix{T}
end

struct RefKFVariables{T}
    A_t::Vector{Matrix{T}}
    B_t::Vector{Matrix{T}}
    D_t::Vector{Matrix{T}}
    G_t::Vector{Matrix{T}}
    I_z_t::Vector{Matrix{T}}
    f_t::Matrix{T}
    n_c::Int
    n_p::Int
    n_s::Int
    n_t::Int
    n_u::Int
    n_x::Int
    n_z_t::Vector{T}
end

struct KalmanData{T}
    observed::ObservedData{T}
    pricing::PricingData{T}
    psi0::Psi0Data{T}
    refkf::RefKFVariables{T}
end

# === 1) Load raw MATLAB data into Float64 structs ===
function load64(data_folder::String)
    # Read all variables
    vars = Dict{Symbol,Any}()
    for fn in ("observedData.mat","pricingData.mat","psi_0.mat","refKFVariables.mat")
        matopen(joinpath(data_folder,fn)) do mf
            for nm in names(mf)
                vars[Symbol(nm)] = read(mf, nm)
            end
        end
    end

    # Flatten zAll cell array (5030×1) into numeric 5030×3670 matrix
    raw_z = vec(vars[:zAll])
    zAll64 = reduce(vcat, [Float64.(m) for m in raw_z])

    # ObservedData
    obs = ObservedData{Float64}(
        Float64.(vec(vars[:ecbRatechangeDates])),
        zAll64,
        Float64.(vec(vars[:times]))
    )

    # PricingData: flatten cell arrays for idContracts, oIndAll, tcAll
    idc = vec(vars[:idContracts])
    oind = vec(vars[:oIndAll])
    tcc = vec(vars[:tcAll])
    TAll64  = [Float64.(m) for m in vec(vars[:TAll])]
    T0All64 = [Float64.(m) for m in vec(vars[:T0All])]
    prc = PricingData{Float64}(
        Float64.(vec(vars[:firstDates])),
        idc,
        TAll64,
        T0All64,
        oind,
        tcc,
        Float64.(vec(vars[:tradeDates]))
    )

    # Psi0Data
    psi = Psi0Data{Float64}(
        Float64.(vars[:Sigma_v]),
        Float64.(vars[:Sigma_w]),
        Float64.(vars[:Sigma_x]),
        Float64.(vec(vars[:a_x])),
        Float64.(vec(vars[:theta_F])),
        Float64.(vars[:theta_g])
    )

    # RefKFVariables
    A64 = [Float64.(m) for m in vec(vars[:A_t])]
    B64 = [Float64.(m) for m in vec(vars[:B_t])]
    D64 = [Float64.(m) for m in vec(vars[:D_t])]
    G64 = [Float64.(m) for m in vec(vars[:G_t])]
    I64 = [Float64.(m) for m in vec(vars[:I_z_t])]
    refkf = RefKFVariables{Float64}(
        A64, B64, D64, G64, I64,
        Float64.(vars[:f_t]),
        Int(vars[:n_c]), Int(vars[:n_p]), Int(vars[:n_s]),
        Int(vars[:n_t]), Int(vars[:n_u]), Int(vars[:n_x]),
        Float64.(vec(vars[:n_z_t]))
    )

    return KalmanData{Float64}(obs, prc, psi, refkf)
end

# === 2) Convert Float64 → Float32 ===
function convert_to_f32(kd::KalmanData{Float64})
    obs = ObservedData{Float32}(
        Float32.(kd.observed.ecbRatechangeDates),
        Float32.(kd.observed.zAll),
        Float32.(kd.observed.times)
    )
    prc = PricingData{Float32}(
        Float32.(kd.pricing.firstDates),
        kd.pricing.idContracts,
        [Float32.(m) for m in kd.pricing.TAll],
        [Float32.(m) for m in kd.pricing.T0All],
        kd.pricing.oIndAll,
        kd.pricing.tcAll,
        Float32.(kd.pricing.tradeDates)
    )
    psi = Psi0Data{Float32}(
        Float32.(kd.psi0.Sigma_v),
        Float32.(kd.psi0.Sigma_w),
        Float32.(kd.psi0.Sigma_x),
        Float32.(kd.psi0.a_x),
        Float32.(kd.psi0.theta_F),
        Float32.(kd.psi0.theta_g)
    )
    refkf = RefKFVariables{Float32}(
        [Float32.(m) for m in kd.refkf.A_t],
        [Float32.(m) for m in kd.refkf.B_t],
        [Float32.(m) for m in kd.refkf.D_t],
        [Float32.(m) for m in kd.refkf.G_t],
        [Float32.(m) for m in kd.refkf.I_z_t],
        Float32.(kd.refkf.f_t),
        kd.refkf.n_c, kd.refkf.n_p, kd.refkf.n_s,
        kd.refkf.n_t, kd.refkf.n_u, kd.refkf.n_x,
        Float32.(kd.refkf.n_z_t)
    )
    return KalmanData{Float32}(obs, prc, psi, refkf)
end

# === 3) Main run: choose precision ===
"""
    run(data_folder::String; T::Type{<:AbstractFloat}=Float64)

Load .mat files and return KalmanData{T}.
"""
function run(data_folder::String; T::Type{<:AbstractFloat}=Float64)
    kd64 = load64(data_folder)
    return T === Float32 ? convert_to_f32(kd64) : kd64
end

function split_data(kd::KalmanData{T}, ratio::Float64) where {T}
    # number of time points
    n_total   = size(kd.observed.zAll, 1)
    split_idx = Int(floor(ratio * n_total))

    in_obs = ObservedData{T}(
      kd.observed.ecbRatechangeDates,
      kd.observed.zAll[1:split_idx, :],
      kd.observed.times[1:split_idx]
    )
    out_obs = ObservedData{T}(
      kd.observed.ecbRatechangeDates,
      kd.observed.zAll[split_idx+1:end, :],
      kd.observed.times[split_idx+1:end]
    )

    in_prc = PricingData{T}(
      kd.pricing.firstDates,
      kd.pricing.idContracts,
      kd.pricing.TAll[1:split_idx],
      kd.pricing.T0All[1:split_idx],
      kd.pricing.oIndAll[1:split_idx],
      kd.pricing.tcAll[1:split_idx],
      kd.pricing.tradeDates[1:split_idx]
    )
    out_prc = PricingData{T}(
      kd.pricing.firstDates,
      kd.pricing.idContracts,
      kd.pricing.TAll[split_idx+1:end],
      kd.pricing.T0All[split_idx+1:end],
      kd.pricing.oIndAll[split_idx+1:end],
      kd.pricing.tcAll[split_idx+1:end],
      kd.pricing.tradeDates[split_idx+1:end]
    )

    # psi0 is time-independent so stays the same
    in_psi = out_psi = kd.psi0

    in_reff = RefKFVariables{T}(
      kd.refkf.A_t[1:split_idx],
      kd.refkf.B_t[1:split_idx],
      kd.refkf.D_t[1:split_idx],
      kd.refkf.G_t[1:split_idx],
      kd.refkf.I_z_t[1:split_idx],
      kd.refkf.f_t[1:split_idx, :],
      kd.refkf.n_c, kd.refkf.n_p, kd.refkf.n_s,
      split_idx,
      kd.refkf.n_u, kd.refkf.n_x,
      kd.refkf.n_z_t[1:split_idx]
    )
    out_reff = RefKFVariables{T}(
      kd.refkf.A_t[split_idx+1:end],
      kd.refkf.B_t[split_idx+1:end],
      kd.refkf.D_t[split_idx+1:end],
      kd.refkf.G_t[split_idx+1:end],
      kd.refkf.I_z_t[split_idx+1:end],
      kd.refkf.f_t[split_idx+1:end, :],
      kd.refkf.n_c, kd.refkf.n_p, kd.refkf.n_s,
      n_total - split_idx,
      kd.refkf.n_u, kd.refkf.n_x,
      kd.refkf.n_z_t[split_idx+1:end]
    )

    return (
      insample  = KalmanData{T}(in_obs,  in_prc, in_psi, in_reff),
      outsample = KalmanData{T}(out_obs, out_prc,out_psi,out_reff)
    )
end

end # module
