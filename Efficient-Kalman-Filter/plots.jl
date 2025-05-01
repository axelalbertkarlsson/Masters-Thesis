module plots

using Plots
using Dates
gr()

function matlab2date(serial::Real)
    # Convert MATLAB serial to Julia Date (MATLAB starts on 0000-01-00, Julia on 0001-01-01)
    return Date("0000-01-01") + Day(round(Int, serial))
end

function plot3DCurve(times, fAll)
    maturities = (1:size(fAll, 1)) ./ 365
    fMatrix = fAll'

    # Convert serial dates to actual Date format
    dates = matlab2date.(times)

    # Plot directly without meshgrid: use surface(xvec, yvec, zmatrix)
    surface(
        maturities, dates, fMatrix;
        # xlabel = nothing,
        # ylabel = nothing,
        # zlabel = nothing,
        title = "Synthetic Forward Rate Surface",
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
    
    
    
end

end # module