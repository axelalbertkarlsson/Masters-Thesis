using Revise
include("loadData.jl")
include("pricingFunctions.jl")

using .loadData
using .pricingFunctions

# Load data
data = loadData.run()

# Initialize 
oAll = [zeros(103, 22) for _ in 1:data.n_t]

# Calculate oAll
for t in 1:Int(data.n_t)
    oAll[t] = pricingFunctions.calcO(
        data.firstDates[t],
        data.tradeDates[t],
        data.theta_g,
        data.ecbRatechangeDates,
        data.n_c,
        data.n_z_t[t],
        data.T0All[t],
        data.TAll[t]
    )
end
