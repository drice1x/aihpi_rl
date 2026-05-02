#!/usr/bin/env python3
# 05_plot_surface_frontier.py
#
# Produces Chinchilla-style Figure (Approach 3 analogue):
# Left: contours of Rhat(Cu, Cr) in log space + efficient frontier
# Right (optional): isoFLOP slice curves implied by the surface

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = "iso_alloc_plane_runs_total.csv"
PARAMS = "surface_fit_params.csv"
OUTDIR = "figures_spot_parametric_surface"
os.makedirs(OUTDIR, exist_ok=True)

METRIC = "total_reward_mean"
EPS = 1e-12

def prettify(ax):
    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", alpha=0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def Rhat(Cu, Cr, Rinf, A, B, alpha, beta):
    Cu = np.maximum(Cu, EPS)
    Cr = np.maximum(Cr, EPS)
    return Rinf - A / (Cu ** alpha) - B / (Cr ** beta)

def main():
    df = pd.read_csv(DATA)
    p = pd.read_csv(PARAMS).iloc[0]

    for c in ["rollout_flops", "update_flops", METRIC]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["rollout_flops", "update_flops", METRIC])
    df = df[(df["rollout_flops"] > 0) & (df["update_flops"] > 0)]

    Cr = df["rollout_flops"].to_numpy(float)
    Cu = df["update_flops"].to_numpy(float)
    y  = df[METRIC].to_numpy(float)

    Rinf = float(p["Rinf"])
    A = float(p["A"])
    B = float(p["B"])
    alpha = float(p["alpha"])
    beta  = float(p["beta"])

    # Grid ranges based on observed data (log space)
    x_min, x_max = np.log10(Cr.min()), np.log10(Cr.max())
    y_min, y_max = np.log10(Cu.min()), np.log10(Cu.max())

    # Expand a bit for nicer contours
    pad = 0.15
    x_min -= pad; x_max += pad
    y_min -= pad; y_max += pad

    xs = np.linspace(x_min, x_max, 160)
    ys = np.linspace(y_min, y_max, 160)
    X, Y = np.meshgrid(xs, ys)

    Crg = 10**X
    Cug = 10**Y
    Z = Rhat(Cug, Crg, Rinf, A, B, alpha, beta)

    # Efficient frontier: for each total compute C, maximize Rhat(rho C, (1-rho) C)
    C_levels = np.logspace(np.log10(min(Cr.min()+Cu.min(), Cr.min()+Cu.max())),
                           np.log10(max(Cr.max()+Cu.max(), Cr.max()+Cu.min())),
                           40)

    rho_grid = np.linspace(0.02, 0.98, 800)
    Cr_star = []
    Cu_star = []
    for C in C_levels:
        Cu_cand = rho_grid * C
        Cr_cand = (1 - rho_grid) * C
        vals = Rhat(Cu_cand, Cr_cand, Rinf, A, B, alpha, beta)
        j = int(np.nanargmax(vals))
        Cu_star.append(Cu_cand[j])
        Cr_star.append(Cr_cand[j])
    Cu_star = np.array(Cu_star)
    Cr_star = np.array(Cr_star)

    # Plot: contour + data + iso-C diagonals + frontier
    fig, ax = plt.subplots(figsize=(7.0, 5.4))

    # contours
    levels = np.linspace(np.nanpercentile(Z, 10), np.nanpercentile(Z, 95), 10)
    cs = ax.contour(X, Y, Z, levels=levels, linewidths=1.0, alpha=0.9)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    # scatter actual points (in log space)
    ax.scatter(np.log10(Cr), np.log10(Cu), s=45, alpha=0.85)

    # iso-total-compute diagonals: log(Cu) vs log(Cr) doesn’t give straight lines for Cu+Cr=C
    # but we can still overlay curves in log space.
    for C in np.logspace(np.log10(Cr.min()+Cu.min()), np.log10(Cr.max()+Cu.max()), 4):
        rho = np.linspace(0.02, 0.98, 200)
        Cu_iso = rho * C
        Cr_iso = (1-rho) * C
        ax.plot(np.log10(Cr_iso), np.log10(Cu_iso), linestyle="--", linewidth=1.0, alpha=0.6)

    # efficient frontier
    ax.plot(np.log10(Cr_star), np.log10(Cu_star), linewidth=2.0, alpha=0.95, label="Efficient frontier")

    ax.set_xlabel(r"$\log_{10}(\mathrm{FLOPs}_{rollout})$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{FLOPs}_{update})$")
    ax.set_title("Parametric reward surface and efficient frontier")
    prettify(ax)
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "surface_contours_frontier.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "surface_contours_frontier.pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {OUTDIR}/surface_contours_frontier.(png/pdf)")

if __name__ == "__main__":
    main()
