#!/usr/bin/env python3
import numpy as np
import pandas as pd

INCSV = "analysis_table2.csv"
OUT_REC = "recommendations2.csv"
OUT_CV  = "cv_report2.csv"

# choose which metric to optimize
METRICS = ["gsm8k_accuracy", "total_reward_mean"]  # script will skip missing

# for grid-searching optimum
N_GRID = np.logspace(np.log10(0.45), np.log10(8.0), 120)     # model size in B
RHO_GRID = np.logspace(np.log10(0.005), np.log10(0.30), 160) # update fraction range

def design_matrix(logN, logrho):
    return np.column_stack([
        logN**2,
        logrho**2,
        logN*logrho,
        logN,
        logrho,
        np.ones_like(logN),
    ])

def fit_ls(X, y):
    # Least squares with NaN guard
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X2 = X[m]; y2 = y[m]
    if len(y2) < 8:
        return None
    beta, *_ = np.linalg.lstsq(X2, y2, rcond=None)
    return beta

def predict(beta, logN, logrho):
    X = design_matrix(logN, logrho)
    return X @ beta

def loocv_rmse(X, y):
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[m]; y = y[m]
    n = len(y)
    if n < 10:
        return np.nan
    errs = []
    for i in range(n):
        Xm = np.delete(X, i, axis=0)
        ym = np.delete(y, i, axis=0)
        beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        yhat = X[i] @ beta
        errs.append((yhat - y[i])**2)
    return float(np.sqrt(np.mean(errs)))

def main():
    df = pd.read_csv(INCSV)
    out_recs = []
    out_cv = []

    for metric in METRICS:
        if metric not in df.columns:
            continue

        for band, g in df.groupby("band_logC"):
            gg = g.dropna(subset=[metric, "logN", "logrho"]).copy()
            if len(gg) < 10:
                continue

            X = design_matrix(gg["logN"].to_numpy(), gg["logrho"].to_numpy())
            y = gg[metric].to_numpy()

            beta = fit_ls(X, y)
            if beta is None:
                continue

            rmse = loocv_rmse(X, y)

            # grid search optimum (max y)
            logN_grid = np.log10(N_GRID)
            logrho_grid = np.log10(RHO_GRID)
            LN, LR = np.meshgrid(logN_grid, logrho_grid, indexing="xy")
            yhat = predict(beta, LN.ravel(), LR.ravel())
            j = int(np.nanargmax(yhat))
            best_logN = float(LN.ravel()[j])
            best_logrho = float(LR.ravel()[j])

            best_N = 10**best_logN
            best_rho = 10**best_logrho
            best_pred = float(yhat[j])

            # also report nearest *observed* config in that band (practical pick)
            gg["dist"] = (gg["logN"] - best_logN)**2 + (gg["logrho"] - best_logrho)**2
            best_obs = gg.loc[gg["dist"].idxmin()]

            out_recs.append({
                "metric": metric,
                "band_logC": band,
                "band_total_flops_approx": 10**band,
                "fit_rmse_loocv": rmse,
                "pred_opt_model_size_B": best_N,
                "pred_opt_update_fraction": best_rho,
                "pred_opt_metric": best_pred,

                # nearest observed config (the one you can actually run today)
                "nearest_run": best_obs.get("run", ""),
                "nearest_model_size_B": best_obs["model_size_B"],
                "nearest_update_fraction": best_obs.get("rho", np.nan),
                "nearest_metric": best_obs[metric],
                "nearest_steps": best_obs.get("steps", np.nan),
                "nearest_K": best_obs.get("K", np.nan),
                "nearest_L": best_obs.get("L", np.nan),
                "nearest_lora_r": best_obs.get("lora_r", np.nan),
            })

            out_cv.append({
                "metric": metric,
                "band_logC": band,
                "n_points": len(gg),
                "rmse_loocv": rmse,
                "beta_a_logN2": beta[0],
                "beta_b_logrho2": beta[1],
                "beta_c_cross": beta[2],
                "beta_d_logN": beta[3],
                "beta_e_logrho": beta[4],
                "beta_f_const": beta[5],
            })

    rec_df = pd.DataFrame(out_recs).sort_values(["metric","band_logC"])
    cv_df  = pd.DataFrame(out_cv).sort_values(["metric","band_logC"])

    rec_df.to_csv(OUT_REC, index=False)
    cv_df.to_csv(OUT_CV, index=False)

    print(f"[Saved] {OUT_REC}")
    print(f"[Saved] {OUT_CV}")
    print("\nTop recommendations:")
    cols = ["metric","band_logC","pred_opt_model_size_B","pred_opt_update_fraction","pred_opt_metric",
            "nearest_model_size_B","nearest_update_fraction","nearest_metric","nearest_steps","nearest_K","nearest_L","nearest_lora_r"]
    print(rec_df[cols].to_string(index=False))

if __name__ == "__main__":
    main()
