using Revise, LinearAlgebra, Plots, DataFrames, CSV, Statistics

include("loadData.jl")
include("pricingFunctions.jl")
include("newtonMethod.jl")
include("outputData.jl")
include("plots.jl")
include("EKF.jl")

using .EKF
using .plots
using .outputData
using .newtonMethod
using .loadData
using .pricingFunctions

# Clears terminal
clear() = print("\e[2J\e[H")

# Split data: p% in-sample, (1-p)% out-of-sample
p = 0.001

Float32_bool = false

if (!Float32_bool)
    # load as Float64 
    println("Data in Float64")
    data = loadData.run("Efficient-Kalman-Filter/Data")               # Float64 data
    split = loadData.split_data(data, p)
    data_insample = split.insample
    data_outsample = split.outsample
else
    # load as Float32
    println("Data in Float32")
    data = loadData.run("Efficient-Kalman-Filter/Data"; T=Float32)   # Float32 data
    split = loadData.split_data(data, p)
    data_insample = split.insample
    data_outsample = split.outsample
end

# Run Filter
x_filt, P_filt, x_smooth, P_smooth, P_lag, oAll, EAll = EKF.kalman_filter_smoother_lag1(
    data_insample.zAll,
    data_insample.oIndAll,
    data_insample.tcAll,
    data_insample.I_z_t,
    data_insample.f_t,
    Int(data_insample.n_c),
    Int(data_insample.n_p),
    Int(data_insample.n_s),
    Int(data_insample.n_t),
    Int(data_insample.n_u),
    Int(data_insample.n_x),
    Int.(data_insample.n_z_t),
    data_insample.A_t,
    data_insample.B_t,
    data_insample.D_t,
    data_insample.G_t,
    data_insample.Sigma_w,
    data_insample.Sigma_v,
    vec(data_insample.a_x),
    data_insample.Sigma_x,
    vec(data_insample.theta_F),
    data_insample.theta_g,
    data_insample.firstDates,
    data_insample.tradeDates,
    data_insample.ecbRatechangeDates,
    data_insample.T0All,
    data_insample.TAll
 )
println("Regular - Kalman Done")
 #Newton Method
x_filt_NM, P_filt_NM, x_smooth_NM, P_smooth_NM, P_lag_NM, oAll_NM, EAll_NM,
a0_NM, Σx_NM, Σw_NM, Σv_NM, θF_NM, θg_NM =
  EKF.NM(
    data_insample.zAll,
    data_insample.oIndAll,
    data_insample.tcAll,
    data_insample.I_z_t,
    data_insample.f_t,            # ← now included
    Int(data_insample.n_c),
    Int(data_insample.n_p),
    Int(data_insample.n_s),
    Int(data_insample.n_t),
    Int(data_insample.n_u),
    Int(data_insample.n_x),
    Int.(data_insample.n_z_t),
    data_insample.A_t,
    data_insample.B_t,
    data_insample.D_t,
    data_insample.G_t,
    data_insample.firstDates,
    data_insample.tradeDates,
    data_insample.ecbRatechangeDates,
    data_insample.T0All,
    data_insample.TAll,
    data_insample.a_x,            # Vector
    data_insample.Sigma_x,        # Matrix
    data_insample.Sigma_w,        # Matrix
    data_insample.Sigma_v,        # Matrix
    data_insample.theta_F,        # Vector
    data_insample.theta_g;        # Matrix
    tol=1e-6,
    maxiter=1000,
    verbose=true,
    Newton_bool=false, #Determines if Newton then true 
    HF_bool=false, #Determines if Newton
    θg_bool=false,  #Detemines if too include theta_g
  )
  println("NM - Kalman Done")

# Calculate Forward Rates and Repricing
fAll, priceAll, innovationAll = outputData.calculateRateAndRepricing(
    EAll,
    data_insample.zAll,
    data_insample.I_z_t,
    x_smooth,
    oAll,
    data_insample.oIndAll,
    data_insample.tcAll,
    data_insample.theta_g,
    Int.(data_insample.n_z_t),
    Int(data_insample.n_t),
    Int(data_insample.n_s),
    Int(data_insample.n_u),
);
println("Regular - Forward Curve Calculated")
fAll_NM, priceAll_NM, innovationAll_NM = outputData.calculateRateAndRepricing(
    EAll_NM,
    data_insample.zAll,
    data_insample.I_z_t,
    x_smooth_NM,
    oAll_NM,
    data_insample.oIndAll,
    data_insample.tcAll,
    θg_NM, # Kanske ska vara data_insample.theta_g, men känns onajs.
    Int.(data_insample.n_z_t),
    Int(data_insample.n_t),
    Int(data_insample.n_s),
    Int(data_insample.n_u),
);
println("NM - Forward Curve Calculated")

#Sample more time points (e.g., 10 evenly spaced)
n = length(innovationAll)
n_samples = 10
ts = round.(Int, range(1, n, length=n_samples))

### === Write Regular (non-Newton) results === ###
regular_params = [
    "a0" => vec(data_insample.a_x),
    "Σx" => data_insample.Sigma_x,
    "Σw" => data_insample.Sigma_w,
    "Σv" => data_insample.Sigma_v,
    "θF" => vec(data_insample.theta_F),
    "θg" => data_insample.theta_g,
]

open("regular.csv", "w") do io
    println(io, "Regular Parameters")
    for (label, param) in regular_params
        println(io, label)
        show(IOContext(io, :limit => false), "text/plain", param)
        println(io, "\n")
    end

    println(io, "Average Innovations at Selected Time Points (t, mean_innovation)")
    for t in ts
        i = vec(innovationAll[t, :])
        mean_i = mean(i)
        println(io, "t = $t, mean_innovation = $mean_i")
    end
end
println("Regular - Data CSV Done")

### === Write Newton Method results === ###
nm_params = [
    "a0_NM" => a0_NM,
    "Σx_NM" => Σx_NM,
    "Σw_NM" => Σw_NM,
    "Σv_NM" => Σv_NM,
    "θF_NM" => θF_NM,
    "θg_NM" => θg_NM,
]

open("NM.csv", "w") do io
    println(io, "Newton Method Parameters")
    for (label, param) in nm_params
        println(io, label)
        show(IOContext(io, :limit => false), "text/plain", param)
        println(io, "\n")
    end

    println(io, "Average Innovations at Selected Time Points (t, mean_innovation_NM)")
    for t in ts
        i = vec(innovationAll_NM[t, :])
        mean_i = mean(i)
        println(io, "t = $t, mean_innovation_NM = $mean_i")
    end
end
println("Newton Method - Data CSV Done")

# Plot Forward Rate Curve (Should be done in Matlab instead)
plt1 = plots.plot3DCurve(data_insample.times, fAll, "Regular")
plt2 = plots.plot3DCurve(data_insample.times, fAll_NM, "Newton Method")

display(plt1)
println("Regular - Plot Done")
display(plt2)
println("NM - Plot Done")
