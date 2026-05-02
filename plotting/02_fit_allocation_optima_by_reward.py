#!/usr/bin/env python3
import numpy as np
import pandas as pd

INCSV = "analysis_reward_ablation.csv"
OUTREC = "allocation_recommendations_by_reward.csv"

# Feasible rho range for selecting an optimum within this ablation grid
# (we pick the better boundary under the fitted model)
RHO_MIN = 0.005
RHO_MAX = 0.30

def fit_linear_logrho(rho, y):
    rho = np.asarray(rho, float)
    y = np.asarray(y, float)
    m = np.isfinite(rho) & (rho > 0) & np.isfinite(y)
    rho = rho[m]; y = y[m]
    if len(y) < 2:
        return None
    x = np.log10(rho)
    a, b = np.polyfit(x, y, deg=1)  # y = a*log10(rho) + b
    return float(a), float(b)

def pred(a, b, rho):
    return a*np.log10(rho) + b

def main():
    df = pd.read_csv(INCSV)

    # numeric coercion
    for c in ["band_logC","rho","update_fraction","logrho","total_reward_mean","lora_r","steps","K","L","total_flops"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ensure rho exists
    if "rho" not in df.columns:
        df["rho"] = df["update_fraction"]

    out = []

    for (rtype, band), g in df.groupby(["reward_type","band_logC"]):
        g = g.dropna(subset=["rho","total_reward_mean"]).copy()
        g = g[g["rho"] > 0]
        if len(g) < 2:
            continue

        fit = fit_linear_logrho(g["rho"].to_numpy(), g["total_reward_mean"].to_numpy())
        if fit is None:
            continue
        a, b = fit

        # Boundary opt under fitted model (since linear has no interior optimum)
        y_lo = pred(a, b, RHO_MIN)
        y_hi = pred(a, b, RHO_MAX)
        if y_hi >= y_lo:
            rho_star = RHO_MAX
            y_star = y_hi
            opt_note = "boundary_high"
        else:
            rho_star = RHO_MIN
            y_star = y_lo
            opt_note = "boundary_low"

        # nearest observed point (practical config)
        g["dist"] = (np.log10(g["rho"]) - np.log10(rho_star))**2
        best = g.loc[g["dist"].idxmin()]

        out.append({
            "reward_type": rtype,
            "band_logC": float(band),
            "band_total_flops_approx": float(10**float(band)),
            "n_points": int(len(g)),

            # main ablation signal:
            "slope_a_logrho": a,   # if positive -> higher rho helps; if negative -> lower rho helps
            "intercept_b": b,

            "pred_opt_update_fraction": float(rho_star),
            "pred_reward": float(y_star),
            "opt_note": opt_note,

            "nearest_rho": float(best["rho"]),
            "nearest_reward": float(best["total_reward_mean"]),
            "nearest_run": best.get("run",""),
            "nearest_lora_r": best.get("lora_r", np.nan),
            "nearest_steps": best.get("steps", np.nan),
            "K": best.get("K", np.nan),
            "L": best.get("L", np.nan),
        })

    if not out:
        raise RuntimeError("No fits produced — check columns and that total_reward_mean is present.")

    rec = pd.DataFrame(out).sort_values(["reward_type","band_logC"])
    rec.to_csv(OUTREC, index=False)

    print(f"[Saved] {OUTREC}\n")
    print(rec[[
        "reward_type","band_logC","n_points",
        "slope_a_logrho","pred_opt_update_fraction","opt_note",
        "nearest_rho","nearest_reward","nearest_lora_r","nearest_steps"
    ]].to_string(index=False))

if __name__ == "__main__":
    main()
