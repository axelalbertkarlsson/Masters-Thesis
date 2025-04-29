using Revise
include("loadData.jl")
include("pricingFunctions.jl")
using .loadData
using .pricingFunctions
using Random, Distributions, Plots
using Dates
using LinearAlgebra

# This will now work without any error
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

# H, u, g, G = pricingFunctions.taylorApprox(oAll[1], data.oIndAll[1], data.tcAll[1], ones(Int(data.n_s),1), data.I_z_t[1], data.n_z_t[1]);
# size(H)
# test can be removed
cov_v_0 = data.G_t[1,1] * data.Sigma_v * data.G_t[1,1]'
cov_w_0 = data.D_t[1,1] * data.Sigma_w * data.D_t[1,1]'
v_0 = rand(MvNormal(zeros(size(cov_v_0, 1)), cov_v_0))
w_0 = rand(MvNormal(zeros(size(cov_w_0, 1)), cov_w_0))

# test of a kalman loop
n = 10

# Static init variables
x_0 = rand(MvNormal(vec(data.a_x), data.Sigma_x))

# noise
cov_v = Vector{Matrix{Float64}}(undef, n)
cov_w = Vector{Matrix{Float64}}(undef, n)
v = Vector{Vector{Float64}}(undef, n)
w = Vector{Vector{Float64}}(undef, n)

# Transition parameters
F = Vector{Matrix{Float64}}(undef, n)
u = Vector{Matrix{Float64}}(undef, n)

# variables
x = Vector{Vector{Float64}}(undef, n)
z = Vector{Vector{Float64}}(undef, n)

theta_F_diag_matrix = Diagonal(vec(data.theta_F))
# Memory tracking arrays
mem_usage = Float64[]
timestamps = DateTime[]

for t in 1:n
    # noise
    cov_v[t] = data.G_t[t,1] * data.Sigma_v * data.G_t[t,1]'
    cov_w[t] = data.D_t[t,1] * data.Sigma_w * data.D_t[t,1]'
    v[t] = rand(MvNormal(zeros(size(cov_v[t], 1)), cov_v[t]))
    w[t] = rand(MvNormal(zeros(size(cov_w[t], 1)), cov_w[t]))

    # Helper equations to get F, H, U
    H, u, g, G = pricingFunctions.taylorApprox(oAll[t], data.oIndAll[t], data.tcAll[t], ones(Int(data.n_s)), data.I_z_t[t], data.n_z_t[t]);
    F[t] = data.A_t[t,1] * theta_F_diag_matrix * data.B_t[t,1]
    x[t] = ones(Int(data.n_s)) #this is temp to check matrix sizes
    z[t] = H * x[t] + u + v[t]
    # u[t] = F


    # calculations to get f
    # theta_F_diag = zeros(size(data.A_t[t,1]))

    # for i in 1:size(data.A_t[t,1], 1)
    #     theta_F_diag[i,i] = data.theta_F[i]
    # end

    # F[t] = data.A_t[t,1] * theta_F_diag * data.B_t[t,1]
    # F[t] = data.A_t[t,1] * Diagonal(data.theta_F) * data.B_t[t,1]
    # F_diag = Diagonal(vec(F[t]))
    # F[t] = data.A_t[t,1] * F_diag * data.B_t[t,1]
end

# a = data.a_x
# thetaF_diag = zeros(size(a))
# for i in 1:min(size(a,1), size(a,2))
#     thetaF_diag[i,i] = data.theta_F[i]
# end

size(data.theta_F)

size(data.A_t[1,1])

size(data.B_t[1,1])
# print(v_t)
# print(w_t)

size(data.G_t[1,1])

# print(data.G_t[1,1])
size(data.D_t)
size(data.D_t)

## TIMING included
# using Revise
# include("loadData.jl")
# using .loadData
# using Random, Distributions, Plots
# using Dates

# # This will now work without any error
# data = loadData.run()

# # println(data.zAll[1,1])
# # println(data.G_t[1][1,2])
# # println(data.idContracts[1])

# # inital variables
# x_0 = rand(MvNormal(vec(data.a_x), data.Sigma_x))

# # test can be removed
# cov_v_0 = data.G_t[1,1] * data.Sigma_v * data.G_t[1,1]'
# cov_w_0 = data.D_t[1,1] * data.Sigma_w * data.D_t[1,1]'
# v_0 = rand(MvNormal(zeros(size(cov_v_0, 1)), cov_v_0))
# w_0 = rand(MvNormal(zeros(size(cov_w_0, 1)), cov_w_0))

# # test of a kalman loop
# n = 100
# cov_v = Vector{Matrix{Float64}}(undef, n)
# cov_w = Vector{Matrix{Float64}}(undef, n)
# v = Vector{Vector{Float64}}(undef, n)
# w = Vector{Vector{Float64}}(undef, n)

# # Memory tracking arrays
# mem_usage = Float64[]
# timestamps = DateTime[]


# for t in 1:n
#     allocated = @allocated begin
#         cov_v[t] = data.G_t[t,1] * data.Sigma_v * data.G_t[t,1]'
#         cov_w[t] = data.D_t[t,1] * data.Sigma_w * data.D_t[t,1]'
#         v[t] = rand(MvNormal(zeros(size(cov_v[t], 1)), cov_v[t]))
#         w[t] = rand(MvNormal(zeros(size(cov_w[t], 1)), cov_w[t]))
#     end
#     # Record memory usage
#     push!(mem_usage, allocated)
#     push!(timestamps, now())

#     # Sleep for 1 ms
#     sleep(0.001)
# end

# # Plotting
# plot_object = plot(1:n, mem_usage, xlabel="Iteration", ylabel="Allocated Bytes", title="Memory Allocation per Iteration", lw=2)
# display(plot_object)
# # print(v_t)
# # print(w_t)

# size(data.G_t[1,1])

# # print(data.G_t[1,1])
# size(data.D_t)
# size(data.D_t)