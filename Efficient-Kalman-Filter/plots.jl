module plots

using Plots
using Dates
using Statistics
gr()

export plot_benchmarks, matlab2date, plot3DCurve

#— Scalar‐alloc dispatch: both nm_alloc & em_alloc scalars —#
function plot_benchmarks(
    nm_times::AbstractVector{<:Real},
    nm_alloc::Number,
    em_times::AbstractVector{<:Real},
    em_alloc::Number;
    figsize=(800,300)
)
    plot_benchmarks(
      nm_times,
      fill(nm_alloc, length(nm_times)),
      em_times,
      fill(em_alloc, length(em_times));
      figsize=figsize
    )
end

#— Mixed dispatch: nm_alloc scalar, em_alloc vector —#
function plot_benchmarks(
    nm_times::AbstractVector{<:Real},
    nm_alloc::Number,
    em_times::AbstractVector{<:Real},
    em_alloc::AbstractVector{<:Real};
    figsize=(800,300)
)
    plot_benchmarks(
      nm_times,
      fill(nm_alloc, length(nm_times)),
      em_times,
      em_alloc;
      figsize=figsize
    )
end

#— Main: both nm_alloc & em_alloc are vectors —#
"""
    plot_benchmarks(nm_times, nm_alloc, em_times, em_alloc; figsize)

Draw *two* separate plots:

1. **Execution time** per iteration (seconds), with the NM/EM averages in the title.  
2. **Memory allocated** per iteration (bytes), with the NM/EM averages in the title.
"""
function plot_benchmarks(
    nm_times::AbstractVector{<:Real},
    nm_alloc::AbstractVector{<:Real},
    em_times::AbstractVector{<:Real},
    em_alloc::AbstractVector{<:Real};
    figsize=(800,300)
)
    # compute averages
    μ_nm_t = mean(nm_times)
    μ_em_t = mean(em_times)
    μ_nm_a = mean(nm_alloc)
    μ_em_a = mean(em_alloc)

    # Plot 1: execution time
    p1 = plot(
      1:length(nm_times), nm_times,
      label="NM", xlabel="Iteration", ylabel="Time (s)",
      title="Exec Time — NM avg=$(round(μ_nm_t,digits=3))s, EM avg=$(round(μ_em_t,digits=3))s",
      size=figsize
    )
    plot!(p1, 1:length(em_times), em_times, label="EM")
    display(p1)

    # Plot 2: memory allocation
    p2 = plot(
      1:length(nm_alloc), nm_alloc,
      label="NM", xlabel="Iteration", ylabel="Bytes",
      title="Memory Alloc — NM avg=$(round(μ_nm_a)) B, EM avg=$(round(μ_em_a)) B",
      size=figsize
    )
    plot!(p2, 1:length(em_alloc), em_alloc, label="EM")
    display(p2)
end

function matlab2date(serial::Real)
    return Date("0000-01-01") + Day(round(Int, serial))
end

function plot3DCurve(times, fAll, subtitle)
    maturities = (1:size(fAll, 1)) ./ 365
    fMatrix = fAll'
    dates = matlab2date.(times)
    return surface(
        maturities, dates, fMatrix;
        title = "Synthetic Forward Rate Surface – " * subtitle,
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
        zlims = (minimum(fAll), maximum(fAll))
    )
end

end # module
