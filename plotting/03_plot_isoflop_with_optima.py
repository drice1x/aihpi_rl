#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = "analysis_table2.csv"
RECS = "recommendations2.csv"
OUTDIR = "figures_spot_fit2"
os.makedirs(OUTDIR, exist_ok=True)

def prettify(ax):
    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", alpha=0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def main():
    df = pd.read_csv(DATA)
    rec = pd.read_csv(RECS)

    for metric in ["gsm8k_accuracy", "total_reward_mean"]:
        if metric not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(6.2, 4.2))

        for band, g in df.groupby("band_logC"):
            gg = g.dropna(subset=[metric, "model_size_B"]).copy()
            if len(gg) < 3:
                continue
            ax.scatter(gg["model_size_B"], gg[metric], s=55, alpha=0.85, label=f"log10(C)≈{band:.2f}")

            # overlay predicted optimum
            rr = rec[(rec["metric"] == metric) & (rec["band_logC"] == band)]
            if len(rr) == 1:
                rrow = rr.iloc[0]
                ax.scatter([rrow["pred_opt_model_size_B"]], [rrow["pred_opt_metric"]],
                           s=140, marker="X", edgecolors="black", linewidths=0.6)
                # annotate with rho*
                ax.annotate(f"ρ*≈{rrow['pred_opt_update_fraction']:.3f}",
                            (rrow["pred_opt_model_size_B"], rrow["pred_opt_metric"]),
                            textcoords="offset points", xytext=(6, 6), fontsize=8)

        ax.set_xscale("log")
        ax.set_xlabel("Model size (B params)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"IsoFLOP curves + fitted optimum (metric={metric})")
        prettify(ax)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"isoflop_optima_{metric}.png"), bbox_inches="tight")
        fig.savefig(os.path.join(OUTDIR, f"isoflop_optima_{metric}.pdf"), bbox_inches="tight")
        plt.close(fig)

    print(f"[Saved] plots to {OUTDIR}/")

if __name__ == "__main__":
    main()
