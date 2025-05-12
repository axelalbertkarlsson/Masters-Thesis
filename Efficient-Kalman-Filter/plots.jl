module plots

using Plots
using Dates
gr()

function matlab2date(serial::Real)
    # Convert MATLAB serial to Julia Date (MATLAB starts on 0000-01-00, Julia on 0001-01-01)
    return Date("0000-01-01") + Day(round(Int, serial))
end

function plot3DCurve(times, fAll, subtitle)
    maturities = (1:size(fAll, 1)) ./ 365
    fMatrix = fAll'

    dates = matlab2date.(times)

    plt = surface(
        maturities, dates, fMatrix;
        title = "Synthetic Forward Rate Surface - " * subtitle,
        color = :jet, 
        camera = (-45, 30),
        cbar = true,
        legend = false,
        linewidth = 0,
        fillalpha = 1.0,
        grid = false,
        xticks = false,
        yticks = false,
        zticks = false,
        size = (900, 600),
        xlims = (minimum(maturities), maximum(maturities)),
        zlims = (minimum(fAll), maximum(fAll)),
    )
    
    return plt
end

"""
    plot_kalman_results(z_pred, z_filt, z_smooth, zAll, label_prefix="")

Given vectors of vectors `z_pred`, `z_filt`, `z_smooth`, and `zAll`, this function:
- Plots the first element of each over time.
- Plots the differences between `zAll` and each of the others (first element).
- Displays all plots.
"""
function plot_kalman_results(z_pred, z_filt, z_smooth, zAll; label_prefix="")
    # Extract the first element from each vector
    z_pred_first_elements = [z_pred[t][1] for t in 1:length(z_pred)]
    z_filt_first_elements = [z_filt[t][1] for t in 1:length(z_filt)]
    z_smooth_first_elements = [z_smooth[t][1] for t in 1:length(z_smooth)]

    # Plot all three: z_pred, z_filt, and z_smooth first elements on the same plot
    plt_all = plot(
        z_pred_first_elements, 
        label="$(label_prefix)z_pred[t][1]", 
        title="First element of z_pred, z_filt, and z_smooth over all time steps", 
        xlabel="Time index", 
        ylabel="Value"
    )
    plot!(plt_all, z_filt_first_elements, label="$(label_prefix)z_filt[t][1]")
    plot!(plt_all, z_smooth_first_elements, label="$(label_prefix)z_smooth[t][1]")
    display(plt_all)

    # Compute the difference between zAll and z_pred, z_filt, z_smooth for the first element at each time step, starting at time 2
    zAll_first_elements = [zAll[t][1] for t in 2:length(zAll)]
    z_pred_first_elements_2 = [z_pred[t][1] for t in 2:length(z_pred)]
    z_filt_first_elements_2 = [z_filt[t][1] for t in 2:length(z_filt)]
    z_smooth_first_elements_2 = [z_smooth[t][1] for t in 2:length(z_smooth)]

    diff_z_pred = [zAll_first_elements[t] - z_pred_first_elements_2[t] for t in 1:length(zAll_first_elements)]
    diff_z_filt = [zAll_first_elements[t] - z_filt_first_elements_2[t] for t in 1:length(zAll_first_elements)]
    diff_z_smooth = [zAll_first_elements[t] - z_smooth_first_elements_2[t] for t in 1:length(zAll_first_elements)]

    # Plot the differences starting at time 2
    plt_diff = plot(
        diff_z_pred,
        label="$(label_prefix)zAll[t][1] - z_pred[t][1]",
        title="Difference: zAll - z_pred/z_filt/z_smooth (first element, t=2:end)",
        xlabel="Time index (starting at t=2)",
        ylabel="Difference"
    )
    plot!(plt_diff, diff_z_filt, label="$(label_prefix)zAll[t][1] - z_filt[t][1]")
    plot!(plt_diff, diff_z_smooth, label="$(label_prefix)zAll[t][1] - z_smooth[t][1]")
    display(plt_diff)

    # Create three separate plots for the differences
    plt_diff_pred = plot(
        diff_z_pred,
        label = "$(label_prefix)zAll[t][1] - z_pred[t][1]",
        title = "Difference: zAll[t][1] - z_pred[t][1] over all time steps",
        xlabel = "Time index",
        ylabel = "Difference"
    )
    display(plt_diff_pred)

    plt_diff_filt = plot(
        diff_z_filt,
        label = "$(label_prefix)zAll[t][1] - z_filt[t][1]",
        title = "Difference: zAll[t][1] - z_filt[t][1] over all time steps",
        xlabel = "Time index",
        ylabel = "Difference"
    )
    display(plt_diff_filt)

    plt_diff_smooth = plot(
        diff_z_smooth,
        label = "$(label_prefix)zAll[t][1] - z_smooth[t][1]",
        title = "Difference: zAll[t][1] - z_smooth[t][1] over all time steps",
        xlabel = "Time index",
        ylabel = "Difference"
    )
    display(plt_diff_smooth)
end

"""
    plot_innovations(innovationAll; hist_range=(-0.0005, 0.0005), bins_all=1000, bins_zoom=3000)

Given a vector of vectors `innovationAll`, this function:
- Plots all innovations as a scatter plot.
- Plots a histogram of all innovations.
- Plots a zoomed-in histogram for values in `hist_range`.
"""
function plot_innovations(innovationAll; hist_range=(-0.0005, 0.0005), bins_all=1000, bins_zoom=3000)
    row_indices = Int[]
    col_indices = Int[]
    col_values = Float64[]

    for (i, row) in enumerate(innovationAll)
        for (j, val) in enumerate(row)
            push!(row_indices, i)
            push!(col_indices, j)
            push!(col_values, val)
        end
    end

    plt_innov_all = scatter(
        row_indices,
        col_values,
        xlabel = "Time Index",
        ylabel = "Innovation",
        title = "All Innovations at Each Time Point",
        legend = false
    )
    display(plt_innov_all)

    plt_innov_hist = histogram(
        col_values,
        bins = bins_all,
        xlabel = "Innovation",
        ylabel = "Frequency",
        title = "Histogram of All Innovations",
        legend = false,
        normalize = true
    )
    display(plt_innov_hist)

    filtered_col_values = [v for v in col_values if hist_range[1] <= v <= hist_range[2]]

    plt_innov_hist_zoom = histogram(
        filtered_col_values,
        bins = bins_zoom,
        xlabel = "Innovation",
        ylabel = "Frequency",
        title = "Histogram of All Innovations (Zoomed)",
        legend = false,
        normalize = true
    )
    display(plt_innov_hist_zoom)
end

end # module
