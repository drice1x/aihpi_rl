#!/usr/bin/env python3
import numpy as np
import pandas as pd

INCSV = "analysis_table2_with_validationruns.csv"
OUT_REC = "allocation_recommendations2.csv"
OUT_FIT = "allocation_fit_coeffs2.csv"

# Metrics to optimize (skip if missing)
METRICS = ["gsm8k_accuracy", "total_reward_mean"]

# We fit y = a*(logrho)^2 + b*(logrho) + c per (band, model_size_B bucket)
# Grid for finding optimum (rho in [0.005, 0.30] is typical in your runs)
RHO_GRID = np.logspace(np.log10(0.005), np.log10(0.30), 400)

def bucket_model_size(ms):
    """Coarse buckets for readability; adjust if you want finer."""
    if ms < 1.0: return "0.5B"
    if ms < 2.2: return "1.5B"
    if ms < 5.0: return "3B"
    return "7B"

def fit_parabola(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(y) < 5:
        return None
    # y = a x^2 + b x + c
    a, b, c = np.polyfit(x, y, deg=2)
    return float(a), float(b), float(c)

def parabola_optimum(a, b):
    if not np.isfinite(a) or abs(a) < 1e-12:
        return None
    return -b / (2*a)

def loocv_rmse(x, y):
    """Leave-one-out RMSE for a quadratic in x."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = len(y)
    if n < 6:
        return np.nan
    errs = []
    for i in range(n):
        xx = np.delete(x, i)
        yy = np.delete(y, i)
        a, b, c = np.polyfit(xx, yy, deg=2)
        yhat = a*x[i]**2 + b*x[i] + c
        errs.append((yhat - y[i])**2)
    return float(np.sqrt(np.mean(errs)))

def main():
    df = pd.read_csv(INCSV)

    # numeric coercion
    for c in ["band_logC","model_size_B","rho","update_fraction","logrho","steps","K","L","lora_r","total_flops"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Accept rho in either column name
    if "rho" not in df.columns:
        if "update_fraction" in df.columns:
            df["rho"] = df["update_fraction"]
        else:
            raise ValueError("Need column rho or update_fraction in analysis table.")

    if "logrho" not in df.columns:
        df["logrho"] = np.log10(df["rho"].where(df["rho"] > 0))

    if "model_size_B" not in df.columns:
        df["model_size_B"] = np.nan

    df["model_bucket"] = df["model_size_B"].apply(lambda x: bucket_model_size(x) if np.isfinite(x) else "unknown")

    out_recs = []
    out_fits = []

    # group by compute band and model bucket (so you can later extend to multi-model)
    for (band, mb), g in df.groupby(["band_logC", "model_bucket"]):
        gg = g.dropna(subset=["logrho"]).copy()
        if len(gg) < 5:
            continue

        for metric in METRICS:
            if metric not in gg.columns:
                continue
            g2 = gg.dropna(subset=[metric]).copy()
            if len(g2) < 5:
                continue

            x = g2["logrho"].to_numpy()
            y = g2[metric].to_numpy()

            coeff = fit_parabola(x, y)
            if coeff is None:
                continue
            a, b, c = coeff

            rmse = loocv_rmse(x, y)

            # Find optimum rho* by grid-searching prediction
            logrho_grid = np.log10(RHO_GRID)
            yhat = a*logrho_grid**2 + b*logrho_grid + c
            j = int(np.nanargmax(yhat))
            logrho_star = float(logrho_grid[j])
            rho_star = float(RHO_GRID[j])
            pred_star = float(yhat[j])

            # nearest observed config (practical)
            g2["dist"] = (g2["logrho"] - logrho_star)**2
            best_obs = g2.loc[g2["dist"].idxmin()]

            out_recs.append({
                "metric": metric,
                "band_logC": float(band),
                "band_total_flops_approx": float(10**float(band)),
                "model_bucket": mb,
                "n_points": int(len(g2)),
                "fit_rmse_loocv": rmse,

                "pred_opt_update_fraction": rho_star,
                "pred_opt_metric": pred_star,

                "nearest_run": best_obs.get("run", ""),
                "nearest_update_fraction": float(best_obs["rho"]),
                "nearest_metric": float(best_obs[metric]),
                "nearest_model_size_B": float(best_obs.get("model_size_B", np.nan)),
                "nearest_steps": best_obs.get("steps", np.nan),
                "nearest_K": best_obs.get("K", np.nan),
                "nearest_L": best_obs.get("L", np.nan),
                "nearest_lora_r": best_obs.get("lora_r", np.nan),
            })

            out_fits.append({
                "metric": metric,
                "band_logC": float(band),
                "model_bucket": mb,
                "n_points": int(len(g2)),
                "rmse_loocv": rmse,
                "a_logrho2": a,
                "b_logrho": b,
                "c_const": c,
            })

    if not out_recs:
        raise RuntimeError(
            "No fits produced. Check that analysis_table2.csv contains rho/logrho "
            "and the metric columns, and that rho > 0."
        )

    rec_df = pd.DataFrame(out_recs).sort_values(["metric", "band_logC", "model_bucket"])
    fit_df = pd.DataFrame(out_fits).sort_values(["metric", "band_logC", "model_bucket"])

    rec_df.to_csv(OUT_REC, index=False)
    fit_df.to_csv(OUT_FIT, index=False)

    print(f"[Saved] {OUT_REC}")
    print(f"[Saved] {OUT_FIT}\n")
    print(rec_df[[
        "metric","band_logC","model_bucket","n_points",
        "pred_opt_update_fraction","pred_opt_metric",
        "nearest_update_fraction","nearest_metric","nearest_lora_r","nearest_steps"
    ]].to_string(index=False))

if __name__ == "__main__":
    main()
