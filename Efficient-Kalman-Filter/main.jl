using Revise, LinearAlgebra, Plots, DataFrames, CSV, Statistics, Printf

# Clears terminal
echo_clear() = print("\e[2J\e[H")

# Load & watch for changes
Revise.includet("loadData.jl")
Revise.includet("pricingFunctions.jl")
Revise.includet("newtonMethod.jl")
Revise.includet("outputData.jl")
Revise.includet("plots.jl")
include("EKF.jl")
using .loadData, .pricingFunctions, .newtonMethod, .outputData, .plots, .EKF

# — Compute in-sample MSE for given ψ tuple
function compute_ins_mse(ψ::NTuple{6,Any}, ins::KalmanData{Float64}, subtitle)
    Σw, Σv, a0, Σx, θF, θg = ψ

    x_f, P_f, x_s, P_s, P_l, oAll, EAll =
      EKF.kalman_filter_smoother_lag1(
        ins.zAll, ins.oIndAll, ins.tcAll, ins.I_z_t, ins.f_t,
        ins.n_c, ins.n_p, ins.n_s, ins.n_t,
        ins.n_u, ins.n_x, Int.(ins.n_z_t),
        ins.A_t, ins.B_t, ins.D_t, ins.G_t,
        Σw, Σv, a0, Σx, θF, θg,
        ins.firstDates, ins.tradeDates,
        ins.ecbRatechangeDates, ins.T0All, ins.TAll
      )
      fAll, priceAll, innov = outputData.calculateRateAndRepricing(
      EAll, ins.zAll, ins.I_z_t, x_s, oAll,
      ins.oIndAll, ins.tcAll, θg,
      Int.(ins.n_z_t), ins.n_t, ins.n_s, ins.n_u
    )
    # Plot Forward Rate Curve (Should be done in Matlab instead)
    plt1 = plots.plot3DCurve(ins.times, fAll, subtitle)
    display(plt1)
    print(subtitle)
    println(" - Plot Done")

    mse, mae = calculateMSE(innov)
    return mse, mae
end

# — Run NM on a single chunk, return new ψ tuple
function nm_on_chunk(ψ::NTuple{6,Any}, outs::KalmanData{Float64}, idxr::UnitRange{Int})
    Σw, Σv, a0, Σx, θF, θg = ψ
    z_c    = outs.zAll[idxr]
    oInd_c = outs.oIndAll[idxr]
    tc_c   = outs.tcAll[idxr]
    Iz_c   = outs.I_z_t[idxr]
    f_c    = outs.f_t[idxr, :]
    nzc    = Int.(outs.n_z_t[idxr])
    A_c, B_c, D_c, G_c = outs.A_t[idxr], outs.B_t[idxr], outs.D_t[idxr], outs.G_t[idxr]
    fd_c, td_c, ecb_c = outs.firstDates[idxr], outs.tradeDates[idxr], outs.ecbRatechangeDates
    T0_c, TC_c        = outs.T0All[idxr], outs.TAll[idxr]
    Tchunk            = length(idxr)

    x_f, P_f, x_s, P_s, P_l, oAll, EAll,
    a0_new, Σx_new, Σw_new, Σv_new, θF_new, θg_new =
      EKF.NM(
        z_c, oInd_c, tc_c, Iz_c, f_c,
        outs.n_c, outs.n_p, outs.n_s, Tchunk,
        outs.n_u, outs.n_x, nzc,
        A_c, B_c, D_c, G_c,
        fd_c, td_c, ecb_c, T0_c, TC_c,
        a0, Σx, Σw, Σv, θF, θg;
        tol=1e5, maxiter=20, verbose=true,
        Newton_bool=false, θg_bool=true
      )
    return (Σw_new, Σv_new, a0_new, Σx_new, θF_new, θg_new)
end

# — Rolling-window NM: update ψ only if it improves full in-sample MSE
function rolling_optimize(ins::KalmanData{Float64}, outs::KalmanData{Float64}, ψ0::NTuple{6,Any})
    ψ = ψ0
    baseline_mse, baseline_mae = compute_ins_mse(ψ, ins, "Regular")
    @printf("Baseline in-sample → MSE = %.5e, MAE = %.5e\n",
            baseline_mse, baseline_mae)

    # chunk size = 1% of total time steps
    total_t = ins.n_t + outs.n_t
    chunk_sz = max(1, floor(Int, 0.03 * total_t)) #3% works on CJ's Mac
    ranges = [s:min(s+chunk_sz-1, ins.n_t) for s in 1:chunk_sz:ins.n_t]

    for (ci, idxr) in enumerate(ranges)
        @printf("\n--- Chunk %d/%d: Days %d–%d ---\n",
                ci, length(ranges), first(idxr), last(idxr))
        # candidate ψ
        ψ_cand = nm_on_chunk(ψ, ins, idxr)
        mse_cand, mae_cand = compute_ins_mse(ψ_cand, ins, "Newton Nr: $ci")
        delta = mse_cand - baseline_mse
        @printf("Old MSE = %.5e, New MSE = %.5e, Δ = %+.5e\n",
                baseline_mse, mse_cand, delta)
        if mse_cand < baseline_mse
            ψ, baseline_mse, baseline_mae = ψ_cand, mse_cand, mae_cand
            println("⇒ Accepted new ψ; updated baseline.")
            break
        else
            println("⇒ Rejected; retained previous ψ.")
        end
    end
    return ψ
end

# === MAIN ===
echo_clear()
println("Loading data...")
data = loadData.run("Efficient-Kalman-Filter/Data")
split = loadData.split_data(data, 0.8)
ins, outs = split.insample, split.outsample

# initial ψ₀ tuple
ψ0 = (
  ins.Sigma_w,
  ins.Sigma_v,
  vec(ins.a_x),
  ins.Sigma_x,
  vec(ins.theta_F),
  ins.theta_g
)

# rolling-window NM
ψ_final = rolling_optimize(ins, outs, ψ0)

# final in-sample comparison
println("\n=== Final in-sample Comparison ===")
@printf("Initial ψ₀ → MSE = %.5e, MAE = %.5e\n", compute_ins_mse(ψ0, outs, "Regular")...)
@printf("Final ψ_final → MSE = %.5e, MAE = %.5e\n", compute_ins_mse(ψ_final, outs, "Final - Newton")...)
# println(compute_ins_mse(ψ_final, outs)[1] < compute_ins_mse(ψ0, outs)[1] ?
#         "NM better MSE" : "Reg better MSE")
# println(compute_ins_mse(ψ_final, outs, )[2] < compute_ins_mse(ψ0, outs)[2] ?
#         "NM better MAE" : "Reg better MAE")
