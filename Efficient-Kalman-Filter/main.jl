using Revise, LinearAlgebra, Plots, DataFrames, CSV, Statistics, Printf, Dates, MAT

 

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

      fAll, zPredAll, innov, _ = outputData.calculateRateAndRepricing(

      EAll, ins.zAll, ins.I_z_t, x_s, oAll,

      ins.oIndAll, ins.tcAll, θg,

      Int.(ins.n_z_t), ins.n_t, ins.n_s, ins.n_u, ins.G_t, Σv, P_f 

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

 

# — Compute in-sample MSE for given ψ tuple with limited data set for plot

function plot_idxr(ψ::NTuple{6,Any}, ins::KalmanData{Float64}, subtitle,  idxr)

    Σw, Σv, a0, Σx, θF, θg = ψ

    x_f, P_f, x_s, P_s, P_l, oAll, EAll =

      EKF.kalman_filter_smoother_lag1(

        ins.zAll[idxr], ins.oIndAll[idxr], ins.tcAll[idxr], ins.I_z_t[idxr], ins.f_t,

        ins.n_c, ins.n_p, ins.n_s, length(idxr),

        ins.n_u, ins.n_x, Int.(ins.n_z_t[idxr]),

        ins.A_t[idxr], ins.B_t[idxr], ins.D_t[idxr], ins.G_t[idxr],

        Σw, Σv, a0, Σx, θF, θg,

        ins.firstDates[idxr], ins.tradeDates[idxr],

        ins.ecbRatechangeDates, ins.T0All[idxr], ins.TAll[idxr]

      )

      fAll, zPredAll, innov, _ = outputData.calculateRateAndRepricing(

      EAll, ins.zAll[idxr], ins.I_z_t[idxr], x_s, oAll,

      ins.oIndAll[idxr], ins.tcAll[idxr], θg,

      Int.(ins.n_z_t[idxr]), length(idxr), ins.n_s, ins.n_u, ins.G_t[idxr], Σv, P_f 

    )

    # Plot Forward Rate Curve (Should be done in Matlab instead)   

    plt = plots.plot3DCurve(ins.times[idxr], fAll, subtitle)

    display(plt)

    print(subtitle)

    println(" - Plot Done")

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

 

  x_f, P_f, x_s, P_s, P_l, oAll, EAll, a0_new, Σx_new, Σw_new, Σv_new, θF_new, θg_new, em_times, em_alloc, em_iters =

    EKF.EM(

      z_c, oInd_c, tc_c, Iz_c, f_c,

      ins.n_c, ins.n_p, ins.n_s, Tchunk,

      ins.n_u, ins.n_x, nzc,

      A_c, B_c, D_c, G_c,

      fd_c, td_c, ecb_c, T0_c, TC_c,

      ψ,

      maxiter=4, tol=1e-3, verbose=true,

      θg_bool=false

    )

  return (Σw_new, Σv_new, a0_new, Σx_new, θF_new, θg_new), em_times, em_alloc, em_iters

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

    a0_new, Σx_new, Σw_new, Σv_new, θF_new, θg_new, nm_times, nm_alloc, nm_iters =

      EKF.NM(

        z_c, oInd_c, tc_c, Iz_c, f_c,

        ins.n_c, ins.n_p, ins.n_s, Tchunk,

        ins.n_u, ins.n_x, nzc,

        A_c, B_c, D_c, G_c,

        fd_c, td_c, ecb_c, T0_c, TC_c,

        a0, Σx, Σw, Σv, θF, θg;

        tol=1e-3, maxiter=40, verbose=true,

        Newton_bool=false, θg_bool=false

      )

    return (Σw_new, Σv_new, a0_new, Σx_new, θF_new, θg_new), nm_times, nm_alloc, nm_iters

end

 

# — Rolling-window NM: update ψ only if it improves full in-sample MSE

function rolling_optimize(ins::KalmanData{Float64}, outs::KalmanData{Float64}, ψ0::NTuple{6,Any})

    nm_times = Float64[]

    nm_alloc = Float64[]

    nm_iters = 0

    em_times = Float64[]

    em_alloc = Float64[]

    em_iters = 0

    ψ = ψ0

    baseline_mse, baseline_mae,_ = compute_ins_mse(ψ, ins, "Regular")

 

    baseline_mse_NM = baseline_mse

    baseline_mae_NM = baseline_mae

 

    baseline_mse_EM = baseline_mse

    baseline_mae_EM = baseline_mae

 

    @printf("Baseline in-sample → MSE = %.5e, MAE = %.5e\n",

            baseline_mse, baseline_mae)

 

    # chunk size = 1% of total time steps

    total_t = ins.n_t + outs.n_t

    chunk_sz = max(1, floor(Int, 0.0513 * total_t)) #3% works on CJ's Mac with theta_g (0.0513 exactly one year)

    ranges = [s:min(s+chunk_sz-1, ins.n_t) for s in 1:chunk_sz:ins.n_t]

 

    ψ_cand_NM = ψ

    ψ_cand_EM = ψ

    ψ_NM = ψ

    ψ_EM = ψ

 

    for (ci, idxr) in enumerate(ranges)

      if ci % 2 != 0

        @printf("\n--- Chunk (Ins) %d/%d: Days %d (%s) – %d (%s) ---\n",

                ci, length(ranges), first(idxr), excel_date_to_datestring(ins.times[first(idxr)]), last(idxr), excel_date_to_datestring(ins.times[last(idxr)]))

        # candidate ψ

        ψ_cand_NM, nm_times, nm_alloc, nm_iters = nm_on_chunk(ψ_NM, ins, idxr)

        ψ_cand_EM, em_times, em_alloc, em_iters = em_on_chunk(ψ_EM, ins, idxr)

        mse_cand_NM, mae_cand_NM, _ = compute_ins_mse(ψ_cand_NM, ins, "No Plot")

        mse_cand_EM, mae_cand_EM, _ = compute_ins_mse(ψ_cand_EM, ins, "No Plot")

        plot_idxr(ψ0, ins, "Regular Nr: $ci",  idxr)

        plot_idxr(ψ_cand_NM, ins, "NM Nr: $ci",  idxr)

        plot_idxr(ψ_cand_EM, ins, "EM Nr: $ci",  idxr)

 

        MLE = "NM"

        delta = mse_cand_NM - baseline_mse_NM

        @printf("Old MSE = %.5e, New MSE (%s) = %.5e, Δ = %+.5e\n",

                baseline_mse_NM, MLE, mse_cand_NM, delta)

        if mse_cand_NM < baseline_mse_NM

            ψ_NM, baseline_mse_NM, baseline_mae_NM = ψ_cand_NM, mse_cand_NM, mae_cand_NM

            println("⇒ Accepted new ψ for ("*MLE*").")

        else

            println("⇒ Rejected; retained previous ψ for ("*MLE*").")

        end

 

        MLE = "EM"

        delta = mse_cand_EM - baseline_mse_EM

        @printf("Old MSE = %.5e, New MSE (%s) = %.5e, Δ = %+.5e\n",

                baseline_mse_EM, MLE, mse_cand_EM, delta)

        if mse_cand_EM < baseline_mse_EM

            ψ_EM, baseline_mse_EM, baseline_mae_EM = ψ_cand_EM, mse_cand_EM, mae_cand_EM

            println("⇒ Accepted new ψ for ("*MLE*").")

        else

            println("⇒ Rejected; retained previous ψ for ("*MLE*").")

        end

 

     

        #plots.plot_benchmarks(nm_times, nm_alloc, em_times, em_alloc)

      else

        @printf("\n--- Chunk (Outs) %d/%d: Days %d (%s) – %d (%s) ---\n",

        ci, length(ranges), first(idxr), excel_date_to_datestring(ins.times[first(idxr)]), last(idxr), excel_date_to_datestring(ins.times[last(idxr)]))

        # candidate ψ

        Σw, Σv, a0, Σx, θF, θg = ψ_cand_NM

 

        x_f, P_f, x_s, P_s, P_l, oAll, EAll =

        EKF.kalman_filter_smoother_lag1(

          ins.zAll[idxr], ins.oIndAll[idxr], ins.tcAll[idxr], ins.I_z_t[idxr], ins.f_t,

          ins.n_c, ins.n_p, ins.n_s, length(idxr),

          ins.n_u, ins.n_x, Int.(ins.n_z_t[idxr]),

          ins.A_t[idxr], ins.B_t[idxr], ins.D_t[idxr], ins.G_t[idxr],

          Σw, Σv, a0, Σx, θF, θg,

          ins.firstDates[idxr], ins.tradeDates[idxr],

          ins.ecbRatechangeDates, ins.T0All[idxr], ins.TAll[idxr]

        )

        fAll_NM, zPredAll_NM, innovationAll_NM, innovation_likelihood_NM = outputData.calculateRateAndRepricing(

        EAll, ins.zAll[idxr], ins.I_z_t[idxr], x_s, oAll,

        ins.oIndAll[idxr], ins.tcAll[idxr], θg,

        Int.(ins.n_z_t[idxr]), length(idxr), ins.n_s, ins.n_u, ins.G_t[idxr], Σv, P_f

      )

        # zPredAll_NM, innovationAll_NM, innovation_likelihood_NM = EKF.calcOutOfSample(

        #   ins.zAll[idxr], ins.oIndAll[idxr], ins.tcAll[idxr], ins.I_z_t[idxr],

        #   ins.n_c, ins.n_s, length(idxr),

        #   ins.n_x, Int.(ins.n_z_t[idxr]),

        #   ins.A_t[idxr], ins.B_t[idxr], ins.D_t[idxr], ins.G_t[idxr],

        #   Σw, Σv, a0, Σx, θF, θg,

        #   ins.firstDates[idxr], ins.tradeDates[idxr],

        #   ins.ecbRatechangeDates, ins.T0All[idxr], ins.TAll[idxr]

        # )

 

        Σw, Σv, a0, Σx, θF, θg = ψ_cand_EM

 

        x_f, P_f, x_s, P_s, P_l, oAll, EAll =

        EKF.kalman_filter_smoother_lag1(

          ins.zAll[idxr], ins.oIndAll[idxr], ins.tcAll[idxr], ins.I_z_t[idxr], ins.f_t,

          ins.n_c, ins.n_p, ins.n_s, length(idxr),

          ins.n_u, ins.n_x, Int.(ins.n_z_t[idxr]),

          ins.A_t[idxr], ins.B_t[idxr], ins.D_t[idxr], ins.G_t[idxr],

          Σw, Σv, a0, Σx, θF, θg,

          ins.firstDates[idxr], ins.tradeDates[idxr],

          ins.ecbRatechangeDates, ins.T0All[idxr], ins.TAll[idxr]

        )

        fAll_EM, zPredAll_EM, innovationAll_EM, innovation_likelihood_EM = outputData.calculateRateAndRepricing(

        EAll, ins.zAll[idxr], ins.I_z_t[idxr], x_s, oAll,

        ins.oIndAll[idxr], ins.tcAll[idxr], θg,

        Int.(ins.n_z_t[idxr]), length(idxr), ins.n_s, ins.n_u, ins.G_t[idxr], Σv, P_f

      )

        # zPredAll_EM, innovationAll_EM, innovation_likelihood_EM = EKF.calcOutOfSample(

        #   ins.zAll[idxr], ins.oIndAll[idxr], ins.tcAll[idxr], ins.I_z_t[idxr],

        #   ins.n_c, ins.n_s, length(idxr),

        #   ins.n_x, Int.(ins.n_z_t[idxr]),

        #   ins.A_t[idxr], ins.B_t[idxr], ins.D_t[idxr], ins.G_t[idxr],

        #   Σw, Σv, a0, Σx, θF, θg,

        #   ins.firstDates[idxr], ins.tradeDates[idxr],

        #   ins.ecbRatechangeDates, ins.T0All[idxr], ins.TAll[idxr]

        # )

       

        zPredAll_RKF = ins.zPredAll[idxr]

        innovationAll_RKF = ins.innovationAll[idxr]

 

        filename = excel_date_to_datestring(ins.times[first(idxr)])* "_OOS_" * excel_date_to_datestring(ins.times[last(idxr)])*".mat"

 

        outputData.write_results(

          filename,

          fAll_NM,  zPredAll_NM,   innovationAll_NM,   innovation_likelihood_NM,  nm_times,   nm_alloc,   nm_iters,

          fAll_EM,  zPredAll_EM,   innovationAll_EM,   innovation_likelihood_EM,  em_times,   em_alloc,   em_iters,

          zPredAll_RKF,  innovationAll_RKF,

          ins.times[idxr]

        )

        println("⇒ Wrote to "*filename*"...")       

    end

    Σw_f, Σv_f, a0_f, Σx_f, θF_f, θg_f = ψ_NM

    matwrite("psi_final_NM.mat", Dict(

 

      "Sigma_w"  => Σw_f,

 

      "Sigma_v"  => Σv_f,

 

      "a0"       => a0_f,

 

     "Sigma_x"  => Σx_f,

 

      "theta_F"  => θF_f,

 

      "theta_g"  => θg_f,

 

    ))

 

    println("Saved ψ_final → psi_final_NM.mat")

    Σw_f, Σv_f, a0_f, Σx_f, θF_f, θg_f = ψ_EM

    matwrite("psi_final_EM.mat", Dict(

 

      "Sigma_w"  => Σw_f,

 

      "Sigma_v"  => Σv_f,

 

      "a0"       => a0_f,

 

      "Sigma_x"  => Σx_f,

 

      "theta_F"  => θF_f,

 

      "theta_g"  => θg_f,

 

    ))

 

    println("Saved ψ_final → psi_final_EM.mat")

  end

    return ψ_NM, ψ_EM

end

 

# === MAIN ===

echo_clear()

println("Loading data...")

data = loadData.run(joinpath("Efficient-Kalman-Filter","Data"))

split = loadData.split_data(data, 1.0)

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

ψ_final_NM, ψ_final_EM = rolling_optimize(ins, outs, ψ0)

 

# final in-sample comparison

mse_reg, mae_reg, _ = compute_ins_mse(ψ0, ins, "Regular")

mse_NM, mae_NM, zPredNMAll = compute_ins_mse(ψ_final_NM, ins, "Final (NM)")

mse_EM, mae_EM, zPredEMAll = compute_ins_mse(ψ_final_EM, ins, "Final (EM)")

 

println("\n=== Final out-sample Comparison ===")

@printf("Initial ψ₀ → MSE = %.5e, MAE = %.5e\n", mse_reg, mae_reg)

@printf("Final ψ_final_NM → MSE = %.5e, MAE = %.5e\n", mse_NM, mae_NM)

@printf("Final ψ_final_EM → MSE = %.5e, MAE = %.5e\n", mse_EM, mae_EM)

@printf("RKF → MSE = %.5e, MAE = %.5e\n", outputData.calculateMSE(ins.innovationAll)...)
