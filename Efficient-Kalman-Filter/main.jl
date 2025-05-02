using Revise, LinearAlgebra, Plots

# Only include files if the module hasn't been defined yet
if !isdefined(Main, :loadData);        include("loadData.jl");        end
if !isdefined(Main, :pricingFunctions);include("pricingFunctions.jl");end
if !isdefined(Main, :newtonMethod);    include("newtonMethod.jl");    end
if !isdefined(Main, :outputData);      include("outputData.jl");      end
if !isdefined(Main, :plots);           include("plots.jl");           end
if !isdefined(Main, :EKF);             include("EKF.jl");             end

using .EKF
using .plots
using .outputData
using .newtonMethod
using .loadData
using .pricingFunctions

# Clears terminal
clear() = print("\e[2J\e[H")

# Load full data
data = loadData.run()

# Split data: p% in-sample, (1-p)% out-of-sample
p = 0.8
split = loadData.split_data(data, p)
data_insample = split.insample
data_outsample = split.outsample

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

# Plot Forward Rate Curve (Should be done in Matlab instead)
#plots.plot3DCurve(data_insample.times, fAll)
