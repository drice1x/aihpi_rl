#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = "analysis_table2_with_validationruns.csv"
RECS = "allocation_recommendations2.csv"
FITS = "allocation_fit_coeffs2.csv"
OUTDIR = "figures_spot_allocation_fit2"
os.makedirs(OUTDIR, exist_ok=True)

RHO_GRID = np.logspace(np.log10(0.005), np.log10(0.30), 400)

def prettify(ax):
    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", alpha=0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def main():
    df = pd.read_csv(DATA)
    rec = pd.read_csv(RECS)
    fit = pd.read_csv(FITS)

    # numeric coercion
    for c in ["band_logC","rho","update_fraction","logrho","model_size_B","steps","K","L","lora_r",
              "gsm8k_accuracy","total_reward_mean"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "rho" not in df.columns:
        df["rho"] = df["update_fraction"]
    if "logrho" not in df.columns:
        df["logrho"] = np.log10(df["rho"].where(df["rho"] > 0))

    metrics = [m for m in ["gsm8k_accuracy", "total_reward_mean"] if m in df.columns]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))

        # plot per band
        for band, g in df.groupby("band_logC"):
            gg = g.dropna(subset=["rho", metric]).copy()
            if len(gg) < 5:
                continue

            # scatter
            ax.scatter(gg["rho"], gg[metric], s=55, alpha=0.85, label=f"log10(C)≈{band:.2f}")

            # fitted curve (if present)
            ff = fit[(fit["metric"] == metric) & (fit["band_logC"] == float(band))]
            if len(ff) >= 1:
                # if multiple model buckets exist, just plot each (usually 1 for you)
                for _, row in ff.iterrows():
                    a, b, c = row["a_logrho2"], row["b_logrho"], row["c_const"]
                    logr = np.log10(RHO_GRID)
                    yhat = a*logr**2 + b*logr + c
                    ax.plot(RHO_GRID, yhat, linewidth=1.5, alpha=0.9)

            # predicted optimum marker
            rr = rec[(rec["metric"] == metric) & (rec["band_logC"] == float(band))]
            if len(rr) >= 1:
                for _, rrow in rr.iterrows():
                    ax.scatter([rrow["pred_opt_update_fraction"]], [rrow["pred_opt_metric"]],
                               s=130, marker="X", edgecolors="black", linewidths=0.6)

        ax.set_xscale("log")
        ax.set_xlabel("Update fraction ρ = update_flops / total_flops")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"IsoFLOP allocation curves + fitted optima ({metric})")

        prettify(ax)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"allocation_optima_{metric}.png"), bbox_inches="tight")
        fig.savefig(os.path.join(OUTDIR, f"allocation_optima_{metric}.pdf"), bbox_inches="tight")
        plt.close(fig)

    print(f"[Saved] plots to {OUTDIR}/")

if __name__ == "__main__":
    main()
