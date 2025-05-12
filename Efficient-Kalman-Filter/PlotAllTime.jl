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

# Clears terminal
clear() = print("\e[2J\e[H")

# Split data: p% in-sample, (1-p)% out-of-sample
p = 1.0

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


function read_nm_csv(filename)
    params = Dict{String, Any}()
    avg_innovations = []
    current_label = nothing
    current_param_lines = String[]
    reading_params = false
    reading_avg_innov = false

    open(filename, "r") do io
        for line in eachline(io)
            line = strip(line)
            if isempty(line)
                continue
            end
            if line == "Newton Method Parameters"
                reading_params = true
                reading_avg_innov = false
                continue
            elseif startswith(line, "Average Innovations at Selected Time Points")
                reading_params = false
                reading_avg_innov = true
                continue
            end

            if reading_params
                if occursin("_NM", line) && endswith(line, "_NM")
                    if current_label !== nothing && !isempty(current_param_lines)
                        param_str = join(current_param_lines, "\n")
                        param_val = nothing  # <-- fix
                        try
                            param_val = Meta.parse(param_str) |> eval
                        catch
                            param_val = param_str
                        end
                        params[current_label] = param_val
                    end
                    current_label = line
                    empty!(current_param_lines)
                else
                    push!(current_param_lines, line)
                end
            elseif reading_avg_innov
                m = match(r"t\s*=\s*(\d+),\s*mean_innovation_NM\s*=\s*([eE0-9\.\+\-]+)", line)
                if m !== nothing
                    t = parse(Int, m.captures[1])
                    mean_innov = parse(Float64, m.captures[2])
                    push!(avg_innovations, (t, mean_innov))
                end
            end
        end
        # Save last param
        if current_label !== nothing && !isempty(current_param_lines)
            param_str = join(current_param_lines, "\n")
            param_val = nothing  # <-- fix
            try
                param_val = Meta.parse(param_str) |> eval
            catch
                param_val = param_str
            end
            params[current_label] = param_val
        end
    end
    return params, avg_innovations
end

# Usage:
nm_params_loaded, nm_avg_innov = read_nm_csv("NM.csv")

for k in keys(nm_params_loaded)
    println(k)
end

# println(typeof(a0_NM))

# function parse_vector_matrix(str)
#     # Remove header lines like "50-element Vector{Float64}:" or "50×50 Matrix{Float64}:"
#     lines = split(str, '\n')
#     data_lines = [line for line in lines if occursin(r"^[\s\-\d\.eE\+]+$", line)]
#     # If no data lines found, try to find lines that look like numbers
#     if isempty(data_lines)
#         data_lines = [line for line in lines if occursin(r"^-?\d", strip(line))]
#     end
#     # Parse numbers
#     numbers = [parse(Float64, strip(line)) for line in data_lines if !isempty(strip(line))]
#     # Guess shape from header
#     if occursin("Matrix", str)
#         m = match(r"(\d+)×(\d+)", str)
#         if m !== nothing
#             nrow = parse(Int, m.captures[1])
#             ncol = parse(Int, m.captures[2])
#             return reshape(numbers, nrow, ncol)
#         end
#     end
#     return numbers
# end

a0_NM  = nm_params_loaded["a0_NM"]
Σx_NM  = nm_params_loaded["Σx_NM"]
Σw_NM  = nm_params_loaded["Σw_NM"]
Σv_NM  = nm_params_loaded["Σv_NM"]
θF_NM  = nm_params_loaded["θF_NM"]
θg_NM  = nm_params_loaded["θg_NM"]

# Run Filter
x_pred_NM, P_pred_NM, x_filt_NM, P_filt_NM, x_smooth_NM, P_smooth_NM, P_lag_NM, oAll_NM, EAll_NM = EKF.kalman_filter_smoother_lag1(
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
    Σw_NM,
    Σv_NM,
    a0_NM,
    Σx_NM,
    # vec(θF_NM),
    θF_NM,
    θg_NM,
    data_insample.firstDates,
    data_insample.tradeDates,
    data_insample.ecbRatechangeDates,
    data_insample.T0All,
    data_insample.TAll
 )
println("Newton - Kalman Done")

### Axel to calc things
z_pred, z_filt, z_smooth, z_pred_top, z_pred_bottom, z_filt_top, z_filt_bottom, z_smooth_top, z_smooth_bottom = EKF.function_for_plotting_three_shumway_graphs(
    x_pred_NM, 
    P_pred_NM, 
    x_filt_NM, 
    P_filt_NM, 
    x_smooth_NM, 
    P_smooth_NM, 
    oAll_NM, 
    data_insample.oIndAll, 
    data_insample.tcAll, 
    data_insample.I_z_t, 
    Int.(data_insample.n_z_t), 
    Int(data_insample.n_t), 
    Int(data_insample.n_s)
    )

# Calculate Forward Rates and Repricing
fAll_NM, priceAll_NM, innovationAll_NM = outputData.calculateRateAndRepricing(
    EAll_NM,
    data_insample.zAll,
    data_insample.I_z_t,
    x_smooth_NM,
    oAll_NM,
    data_insample.oIndAll,
    data_insample.tcAll,
    θg_NM,
    Int.(data_insample.n_z_t),
    Int(data_insample.n_t),
    Int(data_insample.n_s),
    Int(data_insample.n_u),
);
println("Newton - Forward Curve Calculated")



## ======== Regular Filter method ============ ###
# Run Filter
# x_filt, P_filt, x_smooth, P_smooth, P_lag, oAll, EAll = EKF.kalman_filter_smoother_lag1(
#     data_insample.zAll,
#     data_insample.oIndAll,
#     data_insample.tcAll,
#     data_insample.I_z_t,
#     data_insample.f_t,
#     Int(data_insample.n_c),
#     Int(data_insample.n_p),
#     Int(data_insample.n_s),
#     Int(data_insample.n_t),
#     Int(data_insample.n_u),
#     Int(data_insample.n_x),
#     Int.(data_insample.n_z_t),
#     data_insample.A_t,
#     data_insample.B_t,
#     data_insample.D_t,
#     data_insample.G_t,
#     data_insample.Sigma_w,
#     data_insample.Sigma_v,
#     #vec(data_insample.a_x),
#     vec(Float64.(zeros(Int(data_insample.n_x),1))),
#     data_insample.Sigma_x,
#     vec(data_insample.theta_F),
#     data_insample.theta_g,
#     data_insample.firstDates,
#     data_insample.tradeDates,
#     data_insample.ecbRatechangeDates,
#     data_insample.T0All,
#     data_insample.TAll
#  )
# println("Regular - Kalman done")

# # Calculate Forward Rates and Repricing
# fAll, priceAll, innovationAll = outputData.calculateRateAndRepricing(
#     EAll,
#     data_insample.zAll,
#     data_insample.I_z_t,
#     x_smooth,
#     oAll,
#     data_insample.oIndAll,
#     data_insample.tcAll,
#     data_insample.theta_g,
#     Int.(data_insample.n_z_t),
#     Int(data_insample.n_t),
#     Int(data_insample.n_s),
#     Int(data_insample.n_u),
# );
# println("Regular - Forward Curve Calculated")

## ======= Acutal plotting ========== ##
plt1 = plots.plot3DCurve(data_insample.times, fAll_NM, "Newton Method")
# plt2 = plots.plot3DCurve(data_insample.times, fAll, "Regular")

# display(plt2)
# println("Regular - Plot Done")
display(plt1)
println("Newton - Plot Done")

# Plotting innovations and other shumway plots
plots.plot_kalman_results(z_pred, z_filt, z_smooth, data_insample.zAll)
plots.plot_innovations(innovationAll_NM)

### ===== Old code now in plots.jl ====== ###
## -------------------------------------- ###

# # Extract the first element from each of the 5030 z_pred, z_filt, and z_smooth vectors
# z_pred_first_elements = [z_pred[t][1] for t in 1:length(z_pred)]
# z_filt_first_elements = [z_filt[t][1] for t in 1:length(z_filt)]
# z_smooth_first_elements = [z_smooth[t][1] for t in 1:length(z_smooth)]

# # Plot all three: z_pred, z_filt, and z_smooth first elements on the same plot
# plt_all = plot(
#     z_pred_first_elements, 
#     label="z_pred[t][1]", 
#     title="First element of z_pred, z_filt, and z_smooth over all time steps", 
#     xlabel="Time index", 
#     ylabel="Value"
# )
# plot!(plt_all, z_filt_first_elements, label="z_filt[t][1]")
# plot!(plt_all, z_smooth_first_elements, label="z_smooth[t][1]")
# display(plt_all)

# # Plot of differences 
# # Compute the difference between data_insample.zAll and z_pred, z_filt, z_smooth for the first element at each time step, starting at time 2
# zAll_first_elements = [data_insample.zAll[t][1] for t in 2:length(data_insample.zAll)]
# z_pred_first_elements_2 = [z_pred[t][1] for t in 2:length(z_pred)]
# z_filt_first_elements_2 = [z_filt[t][1] for t in 2:length(z_filt)]
# z_smooth_first_elements_2 = [z_smooth[t][1] for t in 2:length(z_smooth)]

# diff_z_pred = [zAll_first_elements[t] - z_pred_first_elements_2[t] for t in 1:length(zAll_first_elements)]
# diff_z_filt = [zAll_first_elements[t] - z_filt_first_elements_2[t] for t in 1:length(zAll_first_elements)]
# diff_z_smooth = [zAll_first_elements[t] - z_smooth_first_elements_2[t] for t in 1:length(zAll_first_elements)]

# # Plot the differences starting at time 2
# plt_diff = plot(
#     diff_z_pred,
#     label="zAll[t][1] - z_pred[t][1]",
#     title="Difference: data_insample.zAll - z_pred/z_filt/z_smooth (first element, t=2:end)",
#     xlabel="Time index (starting at t=2)",
#     ylabel="Difference"
# )
# plot!(plt_diff, diff_z_filt, label="zAll[t][1] - z_filt[t][1]")
# plot!(plt_diff, diff_z_smooth, label="zAll[t][1] - z_smooth[t][1]")
# display(plt_diff)

# # Create three separate plots for the first elements of z_pred, z_filt, and z_smooth

# # Create three separate plots for the differences between data_insample.zAll and z_pred, z_filt, z_smooth (first element)
# plt_diff_pred = plot(
#     diff_z_pred,
#     label = "zAll[t][1] - z_pred[t][1]",
#     title = "Difference: zAll[t][1] - z_pred[t][1] over all time steps",
#     xlabel = "Time index",
#     ylabel = "Difference"
# )
# display(plt_diff_pred)

# plt_diff_filt = plot(
#     diff_z_filt,
#     label = "zAll[t][1] - z_filt[t][1]",
#     title = "Difference: zAll[t][1] - z_filt[t][1] over all time steps",
#     xlabel = "Time index",
#     ylabel = "Difference"
# )
# display(plt_diff_filt)

# plt_diff_smooth = plot(
#     diff_z_smooth,
#     label = "zAll[t][1] - z_smooth[t][1]",
#     title = "Difference: zAll[t][1] - z_smooth[t][1] over all time steps",
#     xlabel = "Time index",
#     ylabel = "Difference"
# )
# display(plt_diff_smooth)

# # Assume innovationAll_NM is a Vector of Vectors (jagged array)
# row_indices = Int[]
# col_indices = Int[]
# col_values = Float64[]

# for (i, row) in enumerate(innovationAll_NM)
#     for (j, val) in enumerate(row)
#         push!(row_indices, i)
#         push!(col_indices, j)
#         push!(col_values, val)
#     end
# end

# plt_innov_all = scatter(
#     row_indices,
#     col_values,
#     xlabel = "Time Index",
#     ylabel = "Innovation",
#     title = "All Innovations (NM) at Each Time Point",
#     legend = false
# )
# display(plt_innov_all)

# plt_innov_hist = histogram(
#     col_values,
#     bins = 1000,  # You can adjust the number of bins as needed
#     xlabel = "Innovation",
#     ylabel = "Frequency",
#     title = "Histogram of All Innovations (NM)",
#     legend = false,
#     normalize = true  # Optional: normalize to show probability density
# )
# display(plt_innov_hist)

# # Filter values to only those between -0.0025 and 0.0025
# filtered_col_values = [v for v in col_values if -0.0005 <= v <= 0.0005]

# plt_innov_hist = histogram(
#     filtered_col_values,
#     bins = 3000,  # You can adjust the number of bins as needed
#     xlabel = "Innovation",
#     ylabel = "Frequency",
#     title = "Histogram of All Innovations (NM) [-0.0025, 0.0025]",
#     legend = false,
#     normalize = true  # Optional: normalize to show probability density
# )
# display(plt_innov_hist)

## -------------------------------------- ###