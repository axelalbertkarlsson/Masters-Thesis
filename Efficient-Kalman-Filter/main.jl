using Revise
using .loadData

# This will now work without any error
data = loadData.run()

println(data.zAll[1,1])
println(data.G_t[1][1,2])
println(data.idContracts[1])
