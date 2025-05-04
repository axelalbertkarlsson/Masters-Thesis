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


end # module