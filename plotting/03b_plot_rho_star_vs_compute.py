#!/usr/bin/env python3
# 03b_plot_rho_star_vs_compute.py
#
# Figure 3: Optimal update fraction rho*(C) vs compute.
# Reads allocation_recommendations2.csv produced by 02a_fit_allocation_optima.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RECS = "allocation_recommendations2.csv"   # from your 02a script
OUTDIR = "figures_spot_allocation_fit2"
os.makedirs(OUTDIR, exist_ok=True)

# Choose which metric(s) to plot. Keep reward as main.
METRICS = ["total_reward_mean", "gsm8k_accuracy"]

def prettify(ax):
    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", alpha=0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def main():
    rec = pd.read_csv(RECS)

    # numeric coercion
    for c in ["band_logC", "band_total_flops_approx", "pred_opt_update_fraction",
              "pred_opt_metric", "n_points", "fit_rmse_loocv"]:
        if c in rec.columns:
            rec[c] = pd.to_numeric(rec[c], errors="coerce")

    # if you used model buckets, this will include them. For your clean file, likely one.
    # We'll plot each model_bucket separately if present.
    has_bucket = "model_bucket" in rec.columns

    for metric in [m for m in METRICS if m in rec["metric"].unique()]:
        r = rec[rec["metric"] == metric].dropna(subset=["band_logC", "pred_opt_update_fraction"]).copy()
        if len(r) == 0:
            continue

        fig, ax = plt.subplots(figsize=(6.2, 4.2))

        if has_bucket:
            for mb, g in r.groupby("model_bucket"):
                g = g.sort_values("band_logC")
                ax.plot(g["band_logC"], g["pred_opt_update_fraction"], marker="o", linewidth=1.6, label=str(mb))
        else:
            r = r.sort_values("band_logC")
            ax.plot(r["band_logC"], r["pred_opt_update_fraction"], marker="o", linewidth=1.6)

        ax.set_xlabel(r"log10(total FLOPs) band center")
        ax.set_ylabel(r"Predicted optimum update fraction $\rho^*(C)$")
        ax.set_title(f"Optimal compute allocation vs compute ({metric})")
        ax.set_yscale("log")  # rho typically spans decades; remove if you prefer linear

        prettify(ax)
        if has_bucket:
            ax.legend(frameon=False, loc="best")
        fig.tight_layout()

        fig.savefig(os.path.join(OUTDIR, f"rho_star_vs_compute_{metric}.png"), bbox_inches="tight")
        fig.savefig(os.path.join(OUTDIR, f"rho_star_vs_compute_{metric}.pdf"), bbox_inches="tight")
        plt.close(fig)

    print(f"[Saved] Figure 3 plots to {OUTDIR}/")

if __name__ == "__main__":
    main()
