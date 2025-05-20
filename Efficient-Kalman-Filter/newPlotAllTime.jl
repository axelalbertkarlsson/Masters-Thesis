module newPlotAllTime
using Revise, LinearAlgebra, Plots, DataFrames, CSV, Statistics, Printf, Plots

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

export plot_with_params
# p = 1.0
# # load as Float64 
# println("Data in Float64")
# data = loadData.run("Efficient-Kalman-Filter/Data")               # Float64 data
# split = loadData.split_data(data, p)
# data_insample = split.insample
# data_outsample = split.outsample

function plot_with_params(data_insample, ψ_tuple)
    Σw, Σv, a0, Σx, θF, θg = ψ_tuple

    x_pred, P_pred, x_filt, P_filt, x_smooth, P_smooth, P_lag, oAll, EAll = EKF.kalman_filter_smoother_lag1(
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
        Σw, Σv, a0, Σx, θF, θg,
        data_insample.firstDates,
        data_insample.tradeDates,
        data_insample.ecbRatechangeDates,
        data_insample.T0All,
        data_insample.TAll
    )
    println("Kalman Done")

    # z_pred, z_filt, z_smooth, z_pred_top, z_pred_bottom, z_filt_top, z_filt_bottom, z_smooth_top, z_smooth_bottom = EKF.function_for_plotting_three_shumway_graphs(
    #     x_pred, P_pred, x_filt, P_filt, x_smooth, P_smooth, oAll, data_insample.oIndAll, data_insample.tcAll, data_insample.I_z_t, Int.(data_insample.n_z_t), Int(data_insample.n_t), Int(data_insample.n_s)
    # )

    fAll, priceAll, innovationAll = outputData.calculateRateAndRepricing(
        EAll,
        data_insample.zAll,
        data_insample.I_z_t,
        x_smooth,
        oAll,
        data_insample.oIndAll,
        data_insample.tcAll,
        θg,
        Int.(data_insample.n_z_t),
        Int(data_insample.n_t),
        Int(data_insample.n_s),
        Int(data_insample.n_u),
    )
    println("Forward Curve Calculated")

    plt1 = plots.plot3DCurve(data_insample.times, fAll, "ψ_final Parameters")
    display(plt1)
    println("Plot Done")

    # plots.plot_kalman_results(z_pred, z_filt, z_smooth, data_insample.zAll)
    plots.plot_innovations(innovationAll)
end
end # module