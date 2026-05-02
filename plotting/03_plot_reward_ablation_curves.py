#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = "analysis_reward_ablation.csv"
RECS = "allocation_recommendations_by_reward.csv"
OUTDIR = "figures_reward_ablation"
os.makedirs(OUTDIR, exist_ok=True)

RHO_GRID = np.logspace(np.log10(0.005), np.log10(0.35), 500)

def prettify(ax):
    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", alpha=0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def main():
    df = pd.read_csv(DATA)
    rec = pd.read_csv(RECS)

    for band, gband in df.groupby("band_logC"):
        fig, ax = plt.subplots(figsize=(6.0, 3.8))

        for rtype, g in gband.groupby("reward_type"):
            g = g.dropna(subset=["rho","total_reward_mean"])
            ax.scatter(g["rho"], g["total_reward_mean"], s=55, alpha=0.9, label=f"{rtype}")

            # mark rho*
            rr = rec[(rec["reward_type"] == rtype) & (rec["band_logC"] == float(band))]
            if len(rr) == 1:
                rrow = rr.iloc[0]
                ax.scatter([rrow["rho_star"]], [rrow["pred_reward"]], s=140, marker="X",
                           edgecolors="black", linewidths=0.6)

        ax.set_xscale("log")
        ax.set_xlabel(r"Update fraction $\rho = F_{\mathrm{update}}/F_{\mathrm{total}}$")
        ax.set_ylabel("Mean total reward")
        ax.set_title(rf"Reward ablation IsoFLOP curves at $\log_{{10}}C \approx {band:.2f}$")
        prettify(ax)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"reward_ablation_band_{band:.2f}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(OUTDIR, f"reward_ablation_band_{band:.2f}.png"), bbox_inches="tight")
        plt.close(fig)

    print(f"[Saved] figures to {OUTDIR}/")

if __name__ == "__main__":
    main()
