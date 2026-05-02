#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RECS = "allocation_recommendations_by_reward.csv"
OUTDIR = "figures_reward_ablation"
os.makedirs(OUTDIR, exist_ok=True)

def prettify(ax):
    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", alpha=0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def main():
    rec = pd.read_csv(RECS)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for rtype, g in rec.groupby("reward_type"):
        ax.plot(g["band_total_flops_approx"], g["rho_star"], marker="o", linewidth=2, label=rtype)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total post-training FLOPs (band center)")
    ax.set_ylabel(r"Predicted optimum update fraction $\rho^*(C)$")
    ax.set_title(r"Reward ablation: $\rho^*(C)$ shifts with objective")
    prettify(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "rho_star_vs_compute_by_reward.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "rho_star_vs_compute_by_reward.png"), bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {OUTDIR}/rho_star_vs_compute_by_reward.*")

if __name__ == "__main__":
    main()
