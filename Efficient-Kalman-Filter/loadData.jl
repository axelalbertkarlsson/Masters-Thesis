module loadData

using MAT

export KalmanData, run, split_data

# === 1) Flat, parameterized KalmanData exactly matching your old fields ===
struct KalmanData{T<:AbstractFloat}
    # Observed
    ecbRatechangeDates::Vector{T}
    zAll::Vector{Vector{T}}      # ← one vector per time step
    times::Vector{T}

    # Pricing
    firstDates::Vector{T}
    idContracts::Vector{Any}
    TAll::Vector{Matrix{T}}
    T0All::Vector{Matrix{T}}
    oIndAll::Vector{Any}
    tcAll::Vector{Any}
    tradeDates::Vector{T}

    # Psi0
    Sigma_v::Matrix{T}
    Sigma_w::Matrix{T}
    Sigma_x::Matrix{T}
    a_x::Vector{T}
    theta_F::Vector{T}
    theta_g::Matrix{T}

    # RefKFVariables
    A_t::Vector{Matrix{T}}
    B_t::Vector{Matrix{T}}
    D_t::Vector{Matrix{T}}
    G_t::Vector{Matrix{T}}
    I_z_t::Vector{Matrix{T}}
    f_t::Matrix{T}

    # sizes
    n_c::Int
    n_p::Int
    n_s::Int
    n_t::Int
    n_u::Int
    n_x::Int
    n_z_t::Vector{T}
end

# -----------------------------------------------------------------------------
# 2) load64: read the .mat files and build a KalmanData{Float64}
# -----------------------------------------------------------------------------
function load64(data_folder::String)
    # read everything into a Dict
    vars = Dict{Symbol,Any}()
    for fn in ("observedData.mat","pricingData.mat","psi_0.mat","refKFVariables.mat")
        matopen(joinpath(data_folder,fn)) do f
            for nm in names(f)
                vars[Symbol(nm)] = read(f,nm)
            end
        end
    end

    # 2a) ObservedData.zAll: 5030×1 MATLAB cell → Vector of length- T Float64 vectors
    raw_z = vec(vars[:zAll])                     # Vector of 5030 little arrays
    z64   = [Float64.(vec(m)) for m in raw_z]     # Vector{Vector{Float64}}
    n_t   = length(z64)
    n_z   = Float64.(vec(vars[:n_z_t]))           # Vector of length n_t

    # 2b) pull out everything else as Float64
    ed  = Float64.(vec(vars[:ecbRatechangeDates]))
    ts  = Float64.(vec(vars[:times]))

    fd  = Float64.(vec(vars[:firstDates]))
    idc = vec(vars[:idContracts])
    TA  = [Float64.(m) for m in vec(vars[:TAll])]
    T0  = [Float64.(m) for m in vec(vars[:T0All])]
    oi  = vec(vars[:oIndAll])
    tc  = vec(vars[:tcAll])
    td  = Float64.(vec(vars[:tradeDates]))

    Σv  = Float64.(vars[:Sigma_v])
    Σw  = Float64.(vars[:Sigma_w])
    Σx  = Float64.(vars[:Sigma_x])
    ax  = Float64.(vec(vars[:a_x]))
    θF  = Float64.(vec(vars[:theta_F]))
    θg  = Float64.(vars[:theta_g])

    At  = [Float64.(m) for m in vec(vars[:A_t])]
    Bt  = [Float64.(m) for m in vec(vars[:B_t])]
    Dt  = [Float64.(m) for m in vec(vars[:D_t])]
    Gt  = [Float64.(m) for m in vec(vars[:G_t])]
    Iz  = [Float64.(m) for m in vec(vars[:I_z_t])]

    ft  = Float64.(vars[:f_t])

    nc  = Int(vars[:n_c])
    np  = Int(vars[:n_p])
    ns  = Int(vars[:n_s])
    nu  = Int(vars[:n_u])
    nx  = Int(vars[:n_x])

    return KalmanData{Float64}(
      # Observed
      ed,    z64, ts,
      # Pricing
      fd,    idc, TA, T0, oi, tc, td,
      # Psi0
      Σv, Σw, Σx, ax, θF, θg,
      # RefKF
      At, Bt, Dt, Gt, Iz, ft,
      # sizes
      nc, np, ns, n_t, nu, nx, n_z
    )
end

# -----------------------------------------------------------------------------
# 3) convert_to_f32: downcast every numeric field to Float32
# -----------------------------------------------------------------------------
function convert_to_f32(kd::KalmanData{Float64})
    return KalmanData{Float32}(
      # Observed
      Float32.(kd.ecbRatechangeDates),
      [Float32.(v) for v in kd.zAll],
      Float32.(kd.times),
      # Pricing
      Float32.(kd.firstDates),
      kd.idContracts,
      [Float32.(m) for m in kd.TAll],
      [Float32.(m) for m in kd.T0All],
      kd.oIndAll,
      kd.tcAll,
      Float32.(kd.tradeDates),
      # Psi0
      Float32.(kd.Sigma_v),
      Float32.(kd.Sigma_w),
      Float32.(kd.Sigma_x),
      Float32.(kd.a_x),
      Float32.(kd.theta_F),
      Float32.(kd.theta_g),
      # RefKF
      [Float32.(m) for m in kd.A_t],
      [Float32.(m) for m in kd.B_t],
      [Float32.(m) for m in kd.D_t],
      [Float32.(m) for m in kd.G_t],
      [Float32.(m) for m in kd.I_z_t],
      Float32.(kd.f_t),
      # sizes
      kd.n_c, kd.n_p, kd.n_s, kd.n_t, kd.n_u, kd.n_x,
      Float32.(kd.n_z_t)
    )
end

# -----------------------------------------------------------------------------
# 4) run: your single entrypoint
# -----------------------------------------------------------------------------
"""
    run(data_folder::String; T::Type{<:AbstractFloat}=Float64)

Load all the .mat files and return a `KalmanData{T}`.  
If `T===Float32`, you get a Float32‐downcast version; otherwise Float64.
"""
function run(data_folder::String; T::Type{<:AbstractFloat}=Float64)
    kd64 = load64(data_folder)
    return T===Float32 ? convert_to_f32(kd64) : kd64
end

# -----------------------------------------------------------------------------
# 5) split_data: exactly your old split, returning (insample, outsample)
# -----------------------------------------------------------------------------
function split_data(kd::KalmanData{T}, ratio::Float64) where {T}
    n   = kd.n_t
    idx = Int(floor(ratio * n))

    ins = KalmanData{T}(
      kd.ecbRatechangeDates,
      kd.zAll[1:idx],
      kd.times[1:idx],
      kd.firstDates,
      kd.idContracts,
      kd.TAll[1:idx],
      kd.T0All[1:idx],
      kd.oIndAll[1:idx],
      kd.tcAll[1:idx],
      kd.tradeDates[1:idx],
      kd.Sigma_v, kd.Sigma_w, kd.Sigma_x,
      kd.a_x, kd.theta_F, kd.theta_g,
      kd.A_t[1:idx], kd.B_t[1:idx], kd.D_t[1:idx],
      kd.G_t[1:idx], kd.I_z_t[1:idx],
      kd.f_t[1:idx, :],
      kd.n_c, kd.n_p, kd.n_s, idx, kd.n_u, kd.n_x,
      kd.n_z_t[1:idx]
    )

    out = KalmanData{T}(
      kd.ecbRatechangeDates,
      kd.zAll[idx+1:end],
      kd.times[idx+1:end],
      kd.firstDates,
      kd.idContracts,
      kd.TAll[idx+1:end],
      kd.T0All[idx+1:end],
      kd.oIndAll[idx+1:end],
      kd.tcAll[idx+1:end],
      kd.tradeDates[idx+1:end],
      kd.Sigma_v, kd.Sigma_w, kd.Sigma_x,
      kd.a_x, kd.theta_F, kd.theta_g,
      kd.A_t[idx+1:end], kd.B_t[idx+1:end],
      kd.D_t[idx+1:end], kd.G_t[idx+1:end],
      kd.I_z_t[idx+1:end],
      kd.f_t[idx+1:end, :],
      kd.n_c, kd.n_p, kd.n_s, n-idx, kd.n_u, kd.n_x,
      kd.n_z_t[idx+1:end]
    )

    return (insample=ins, outsample=out)
end

end # module
