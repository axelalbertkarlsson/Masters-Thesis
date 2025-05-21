using Revise, LinearAlgebra, Plots, DataFrames, CSV, Statistics, Printf, Dates

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

function excel_date_to_datestring(x::Real)
  offset = 693960  # empirically determined offset
  base = DateTime(1899, 12, 30)
  dt = base + Dates.Second(round(Int, (x - offset) * 86400))
  return Dates.format(dt, "yyyy-mm-dd")
end



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
      fAll, zPredAll, innov = outputData.calculateRateAndRepricing(
      EAll, ins.zAll, ins.I_z_t, x_s, oAll,
      ins.oIndAll, ins.tcAll, θg,
      Int.(ins.n_z_t), ins.n_t, ins.n_s, ins.n_u
    )
    # Plot Forward Rate Curve (Should be done in Matlab instead)
    if subtitle != "No Plot"
      plt1 = plots.plot3DCurve(ins.times, fAll, subtitle)
      display(plt1)
      print(subtitle)
      println(" - Plot Done")
    end

    mse, mae = calculateMSE(innov)
    return mse, mae, zPredAll
end

# — Run NM on a single chunk, return new ψ tuple
function em_on_chunk(ψ::NTuple{6,Any}, ins::KalmanData{Float64}, idxr::UnitRange{Int})
  Σw, Σv, a0, Σx, θF, θg = ψ
  z_c    = ins.zAll[idxr]
  oInd_c = ins.oIndAll[idxr]
  tc_c   = ins.tcAll[idxr]
  Iz_c   = ins.I_z_t[idxr]
  f_c    = ins.f_t[idxr, :]
  nzc    = Int.(ins.n_z_t[idxr])
  A_c, B_c, D_c, G_c = ins.A_t[idxr], ins.B_t[idxr], ins.D_t[idxr], ins.G_t[idxr]
  fd_c, td_c, ecb_c = ins.firstDates[idxr], ins.tradeDates[idxr], ins.ecbRatechangeDates
  T0_c, TC_c        = ins.T0All[idxr], ins.TAll[idxr]
  Tchunk            = length(idxr)

  x_f, P_f, x_s, P_s, P_l, oAll, EAll, a0_new, Σx_new, Σw_new, Σv_new, θF_new, θg_new =
    EKF.EM(
      z_c, oInd_c, tc_c, Iz_c, f_c,
      ins.n_c, ins.n_p, ins.n_s, Tchunk,
      ins.n_u, ins.n_x, nzc,
      A_c, B_c, D_c, G_c,
      fd_c, td_c, ecb_c, T0_c, TC_c,
      ψ,
      maxiter=5, tol=1e3, verbose=true,
      θg_bool=false
    )
  return (Σw_new, Σv_new, a0_new, Σx_new, θF_new, θg_new)
end

# — Run NM on a single chunk, return new ψ tuple
function nm_on_chunk(ψ::NTuple{6,Any}, ins::KalmanData{Float64}, idxr::UnitRange{Int})
    Σw, Σv, a0, Σx, θF, θg = ψ
    z_c    = ins.zAll[idxr]
    oInd_c = ins.oIndAll[idxr]
    tc_c   = ins.tcAll[idxr]
    Iz_c   = ins.I_z_t[idxr]
    f_c    = ins.f_t[idxr, :]
    nzc    = Int.(ins.n_z_t[idxr])
    A_c, B_c, D_c, G_c = ins.A_t[idxr], ins.B_t[idxr], ins.D_t[idxr], ins.G_t[idxr]
    fd_c, td_c, ecb_c = ins.firstDates[idxr], ins.tradeDates[idxr], ins.ecbRatechangeDates
    T0_c, TC_c        = ins.T0All[idxr], ins.TAll[idxr]
    Tchunk            = length(idxr)

    x_f, P_f, x_s, P_s, P_l, oAll, EAll,
    a0_new, Σx_new, Σw_new, Σv_new, θF_new, θg_new =
      EKF.NM(
        z_c, oInd_c, tc_c, Iz_c, f_c,
        ins.n_c, ins.n_p, ins.n_s, Tchunk,
        ins.n_u, ins.n_x, nzc,
        A_c, B_c, D_c, G_c,
        fd_c, td_c, ecb_c, T0_c, TC_c,
        a0, Σx, Σw, Σv, θF, θg;
        tol=1e3, maxiter=20, verbose=true,
        Newton_bool=false, θg_bool=false
      )
    return (Σw_new, Σv_new, a0_new, Σx_new, θF_new, θg_new)
end

# — Rolling-window NM: update ψ only if it improves full in-sample MSE
function rolling_optimize(ins::KalmanData{Float64}, outs::KalmanData{Float64}, ψ0::NTuple{6,Any})
    ψ = ψ0
    baseline_mse, baseline_mae,_ = compute_ins_mse(ψ, ins, "Regular")
    @printf("Baseline in-sample → MSE = %.5e, MAE = %.5e\n",
            baseline_mse, baseline_mae)

    # chunk size = 1% of total time steps
    total_t = ins.n_t + outs.n_t
    chunk_sz = max(1, floor(Int, 0.0513 * total_t)) #3% works on CJ's Mac with theta_g (0.0513 exactly one year)
    ranges = [s:min(s+chunk_sz-1, ins.n_t) for s in 1:chunk_sz:ins.n_t]

    ψ_cand_NM = ψ
    ψ_cand_EM = ψ

    for (ci, idxr) in enumerate(ranges)
      if ci % 2 != 0
        @printf("\n--- Chunk (Ins) %d/%d: Days %d (%s) – %d (%s) ---\n",
                ci, length(ranges), first(idxr), excel_date_to_datestring(ins.times[first(idxr)]), last(idxr), excel_date_to_datestring(ins.times[last(idxr)])) 
        # candidate ψ
        ψ_cand_NM = nm_on_chunk(ψ, ins, idxr)
        ψ_cand_EM = em_on_chunk(ψ, ins, idxr)
        mse_cand_NM, mae_cand_NM, _ = compute_ins_mse(ψ_cand_NM, ins, "Newton Nr: $ci")
        mse_cand_EM, mae_cand_EM, _ = compute_ins_mse(ψ_cand_EM, ins, "EM Nr: $ci")
        if mse_cand_NM < mse_cand_EM
          mse_cand = mse_cand_NM
          mae_cand = mae_cand_NM
          ψ_cand = ψ_cand_NM
          MLE = "NM"
        else
          mse_cand = mse_cand_EM
          mae_cand = mae_cand_EM
          ψ_cand = ψ_cand_EM
          MLE = "EM"
        end
        delta = mse_cand - baseline_mse
        @printf("Old MSE = %.5e, New MSE (%s) = %.5e, Δ = %+.5e\n",
                baseline_mse, MLE, mse_cand, delta)
        if mse_cand < baseline_mse
            ψ, baseline_mse, baseline_mae = ψ_cand, mse_cand, mae_cand
            println("⇒ Accepted new ψ; updated baseline with ("*MLE*").")
        else
            println("⇒ Rejected; retained previous ψ.")
        end
      else 
        @printf("\n--- Chunk (Outs) %d/%d: Days %d (%s) – %d (%s) ---\n",
        ci, length(ranges), first(idxr), excel_date_to_datestring(ins.times[first(idxr)]), last(idxr), excel_date_to_datestring(ins.times[last(idxr)])) 
        # candidate ψ
        Σw, Σv, a0, Σx, θF, θg = ψ_cand_NM
        zPredAll_NM, innovationAll_NM, innovation_likelihood_NM = EKF.calcOutOfSample(
          ins.zAll[idxr], ins.oIndAll[idxr], ins.tcAll[idxr], ins.I_z_t[idxr],
          ins.n_c, ins.n_s, length(idxr),
          ins.n_x, Int.(ins.n_z_t[idxr]),
          ins.A_t[idxr], ins.B_t[idxr], ins.D_t[idxr], ins.G_t[idxr],
          Σw, Σv, a0, Σx, θF, θg,
          ins.firstDates[idxr], ins.tradeDates[idxr],
          ins.ecbRatechangeDates, ins.T0All[idxr], ins.TAll[idxr]
        )


        Σw, Σv, a0, Σx, θF, θg = ψ_cand_EM
        zPredAll_EM, innovationAll_EM, innovation_likelihood_EM = EKF.calcOutOfSample(
          ins.zAll[idxr], ins.oIndAll[idxr], ins.tcAll[idxr], ins.I_z_t[idxr],
          ins.n_c, ins.n_s, length(idxr),
          ins.n_x, Int.(ins.n_z_t[idxr]),
          ins.A_t[idxr], ins.B_t[idxr], ins.D_t[idxr], ins.G_t[idxr],
          Σw, Σv, a0, Σx, θF, θg,
          ins.firstDates[idxr], ins.tradeDates[idxr],
          ins.ecbRatechangeDates, ins.T0All[idxr], ins.TAll[idxr]
        )
        
        zPredAll_RKF = ins.zPredAll[idxr]
        innovationAll_RKF = ins.innovationAll[idxr]

        filename = excel_date_to_datestring(ins.times[first(idxr)])* "_OOS_" * excel_date_to_datestring(ins.times[last(idxr)])*".mat"

        outputData.write_results(
          filename,
          zPredAll_NM,   innovationAll_NM,   innovation_likelihood_NM,
          zPredAll_EM,   innovationAll_EM,   innovation_likelihood_EM,
          zPredAll_RKF,  innovationAll_RKF
        )
        println("⇒ Wrote to "*filename*"...")        
    end
  end
    return ψ
end

# === MAIN ===
echo_clear()
println("Loading data...")
data = loadData.run(joinpath("Efficient-Kalman-Filter","Data"))
split = loadData.split_data(data, 0.999)
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
mse_reg, mae_reg, _ = compute_ins_mse(ψ0, outs, "Regular")
mse_NM, mae_NM, zPredNMAll = compute_ins_mse(ψ_final, outs, "Final - Newton")

println("\n=== Final out-sample Comparison ===")
@printf("Initial ψ₀ → MSE = %.5e, MAE = %.5e\n", mse_reg, mae_reg)
@printf("Final ψ_final → MSE = %.5e, MAE = %.5e\n", mse_NM, mae_NM)
@printf("RKF → MSE = %.5e, MAE = %.5e\n", outputData.calculateMSE(outs.innovationAll)...)