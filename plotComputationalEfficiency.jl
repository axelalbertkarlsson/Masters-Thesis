using MAT, Plots, Statistics, Colors

# Optional: Choose a backend (GR is fine for most use cases)
gr()

# List of files
mat_files = [
    "2006-08-16_OOS_2007-08-21.mat",
    "2008-09-02_OOS_2009-09-04.mat",
    "2010-09-08_OOS_2011-09-06.mat",
    "2012-09-07_OOS_2013-09-12.mat",
    "2014-09-18_OOS_2015-09-23.mat",
    "2016-09-26_OOS_2017-09-26.mat",
    "2018-10-02_OOS_2019-10-07.mat",
    "2020-10-12_OOS_2021-10-12.mat",
    "2022-10-12_OOS_2023-10-12.mat",
    "2024-10-17_OOS_2025-04-17.mat"
]

# Data containers
alloc_EM = Float64[]
alloc_NM = Float64[]
times_EM = Float64[]
times_NM = Float64[]

# Load and extract
for file in mat_files
    mat_data = matread(file)
    if all(haskey(mat_data, var) for var in ["alloc_EM", "alloc_NM", "times_EM", "times_NM"])
        append!(alloc_EM, vec(mat_data["alloc_EM"]))
        append!(alloc_NM, vec(mat_data["alloc_NM"]))
        append!(times_EM, vec(mat_data["times_EM"]))
        append!(times_NM, vec(mat_data["times_NM"]))
    else
        @warn "Missing variable in $file"
    end
end

# Convert units
alloc_EM_gb = alloc_EM ./ 1e9
alloc_NM_gb = alloc_NM ./ 1e9
times_EM_min = times_EM ./ 60
times_NM_min = times_NM ./ 60

# Compute means
mean_alloc_EM = round(mean(alloc_EM_gb), digits=2)
mean_alloc_NM = round(mean(alloc_NM_gb), digits=2)
mean_time_EM = round(mean(times_EM_min), digits=2)
mean_time_NM = round(mean(times_NM_min), digits=2)

em_color = RGB(52/255, 152/255, 219/255)  # bright blue
nm_color = RGB(12/255, 30/255, 51/255)    # deep navy

# === MEMORY PLOT ===
p1 = plot(
    alloc_EM_gb[1:end-30],
    label="EM",
    xlabel="Iterations",
    ylabel="Memory (GB)",
    title="Memory Allocations (EM vs NM)",
    legend=:right,
    alpha=0.8,
    color=em_color
)
plot!(
    alloc_NM_gb,
    label="NM",
    alpha=0.8,
    color=nm_color
)

# Memory plot stats box — positioned right and lower
x_annot1 = 325
y_base1 = maximum(alloc_EM_gb) * 0.7

annotate!(x_annot1, y_base1 + 8, text("Average Memory Allocation", 10, :black))

annotate!(x_annot1 - 25, y_base1 - 4, text("EM:", 9, :black))
annotate!(x_annot1 + 20, y_base1 - 4, text("$(mean_alloc_EM) GB", 9, em_color))

annotate!(x_annot1 - 25, y_base1 - 12, text("NM:", 9, :black))
annotate!(x_annot1 + 20, y_base1 - 12, text("$(mean_alloc_NM) GB", 9, nm_color))


# === TIME PLOT ===
p2 = plot(
    times_EM_min[30:end],
    label="EM",
    xlabel="Iterations",
    ylabel="Execution time (min)",
    title="Execution Time (EM vs NM)",
    legend=:topright,
    alpha=0.8,
    color=em_color
)
plot!(
    times_NM_min,
    label="NM",
    alpha=0.8,
    color=nm_color
)

# Execution time plot stats box — positioned right and slightly lower
x_annot2 = 340
y_base2 = maximum(times_EM_min) * 0.75

annotate!(x_annot2, y_base2 + 0.8, text("Average Execution Time", 10, :black))
annotate!(x_annot2 - 20, y_base2 - 0.4, text("EM:", 9, :black))
annotate!(x_annot2 + 20, y_base2 - 0.4, text("$(mean_time_EM) min", 9, em_color))

annotate!(x_annot2 - 20, y_base2 - 1.2, text("NM:", 9, :black))
annotate!(x_annot2 + 20, y_base2 - 1.2, text("$(mean_time_NM) min", 9, nm_color))


# Save plots
savefig(p1, "memory_allocations_gb_with_stats.png")
savefig(p2, "execution_times_minutes_with_stats.png")
savefig(p1, "memory_allocations_gb_with_stats.svg")
savefig(p2, "execution_times_minutes_with_stats.svg")
