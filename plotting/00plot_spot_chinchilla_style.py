#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INCSV = "iso_alloc_plane_runs_total.csv"
OUTDIR = "figures_spot200126"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------- Style -----------------------
def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def prettify(ax):
    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", alpha=0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def log10_safe(x):
    x = np.asarray(x, dtype=float)
    x = np.where(x > 0, x, np.nan)
    return np.log10(x)

# ----------------------- Data ------------------------
def load_data():
    df = pd.read_csv(INCSV)

    num_cols = [
        "model_size_B","steps","K","L","lora_r","lora_alpha",
        "total_flops","rollout_flops","update_flops","update_fraction",
        "gsm8k_accuracy",
        "total_reward_mean","total_reward_std",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If total_flops missing, recompute from components
    if "total_flops" in df.columns and "rollout_flops" in df.columns and "update_flops" in df.columns:
        bad = df["total_flops"].isna() | (df["total_flops"] <= 0)
        df.loc[bad, "total_flops"] = df.loc[bad, "rollout_flops"] + df.loc[bad, "update_flops"]

    # Ensure update_fraction exists
    if "update_fraction" not in df.columns and "update_flops" in df.columns and "total_flops" in df.columns:
        df["update_fraction"] = df["update_flops"] / df["total_flops"]

    # Keep only sensible rows
    df = df.dropna(subset=["total_flops","model_size_B"])
    df = df[df["total_flops"] > 0]
    return df

# ------------------- Approach 1 ----------------------
def plot_approach1_compute_scaling(df, ycol="gsm8k_accuracy"):
    """
    Chinchilla Approach 1 analogue:
    - For each model size, plot metric vs total_flops.
    - Also plot an 'upper envelope' across all runs (best metric in log bins).
    """
    d = df.dropna(subset=[ycol, "total_flops", "model_size_B"]).copy()
    if len(d) < 5:
        print("[WARN] Not enough data for Approach 1 plot.")
        return

    # bin model sizes coarsely for grouping
    # (works with your 0.5B / 1.5B / 3B / 7B)
    def bucket_size(ms):
        if ms < 1.0: return "0.5B"
        if ms < 2.2: return "1.5B"
        if ms < 5.0: return "3B"
        return "7B"
    d["size_bucket"] = d["model_size_B"].apply(bucket_size)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for b, g in d.groupby("size_bucket"):
        ax.scatter(g["total_flops"], g[ycol], s=35, alpha=0.85, label=b)

    # Upper envelope across all runs (best y in compute bins)
    logC = log10_safe(d["total_flops"].to_numpy())
    bins = np.linspace(np.nanmin(logC), np.nanmax(logC), 18)
    idx = np.digitize(logC, bins)
    env_x, env_y = [], []
    for k in range(1, len(bins)+1):
        sel = d[idx == k]
        if len(sel) == 0:
            continue
        j = sel[ycol].idxmax()
        env_x.append(d.loc[j, "total_flops"])
        env_y.append(d.loc[j, ycol])
    if len(env_x) >= 2:
        ax.plot(env_x, env_y, linewidth=2.0, alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("Total post-training FLOPs")
    ax.set_ylabel(ycol.replace("_", " "))
    ax.set_title(f"Approach 1 analogue: performance vs compute (by model size)")

    prettify(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"approach1_compute_scaling_{ycol}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, f"approach1_compute_scaling_{ycol}.pdf"), bbox_inches="tight")
    plt.close(fig)

# ------------------- Approach 2 ----------------------
def plot_isoflop_curves(df, ycol="gsm8k_accuracy", target_logC=(16.05, 16.25, 16.45), tol=0.06):
    """
    Chinchilla Approach 2 analogue:
    - For several compute budgets C, plot final metric vs model size.
    - Fit a quadratic in log10(model_size_B) to estimate a peak/valley.
    """
    d = df.dropna(subset=[ycol, "total_flops", "model_size_B"]).copy()
    if len(d) < 5:
        print("[WARN] Not enough data for IsoFLOP curves.")
        return

    d["logC"] = log10_safe(d["total_flops"].to_numpy())
    d["logN"] = log10_safe(d["model_size_B"].to_numpy())

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    for t in target_logC:
        band = d[(d["logC"] >= t - tol) & (d["logC"] <= t + tol)].copy()
        if len(band) < 4:
            continue

        # scatter
        ax.scatter(band["model_size_B"], band[ycol], s=45, alpha=0.85, label=f"log10(C)≈{t:.2f}")

        # quadratic fit in logN (like Chinchilla parabola fit), if enough distinct points
        x = band["logN"].to_numpy()
        y = band[ycol].to_numpy()
        if np.unique(np.round(x, 3)).size >= 3:
            coeff = np.polyfit(x, y, deg=2)  # y = a x^2 + b x + c
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            ys = coeff[0]*xs**2 + coeff[1]*xs + coeff[2]
            ax.plot(10**xs, ys, linewidth=1.5, alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("Model size (B params)")
    ax.set_ylabel(ycol.replace("_", " "))
    ax.set_title("Approach 2 analogue: IsoFLOP curves (metric vs model size at fixed compute)")

    prettify(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"approach2_isoflop_curves_{ycol}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, f"approach2_isoflop_curves_{ycol}.pdf"), bbox_inches="tight")
    plt.close(fig)

# ---------------- Allocation map ----------------------
def plot_allocation_map(df, color_col="gsm8k_accuracy"):
    d = df.dropna(subset=["rollout_flops", "update_flops", "total_flops", color_col]).copy()
    d = d[(d["rollout_flops"] > 0) & (d["update_flops"] > 0)]

    if len(d) < 5:
        print("[WARN] Not enough points for allocation map.")
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.6))

    sc = ax.scatter(
        d["rollout_flops"], d["update_flops"],
        c=d[color_col], s=55, alpha=0.9,
        edgecolors="black", linewidths=0.35
    )

    # Iso-total lines: update = C - rollout (shown in linear, plotted on log axes)
    xmin, xmax = d["rollout_flops"].min(), d["rollout_flops"].max()
    xx = np.logspace(np.log10(xmin), np.log10(xmax), 400)

    # choose budgets from data quantiles (robust)
    budgets = np.quantile(d["total_flops"], [0.1, 0.25, 0.5, 0.75, 0.9])
    budgets = np.unique(np.round(budgets, -int(np.floor(np.log10(budgets.max())))+2))  # light rounding

    for C in budgets:
        yy = C - xx
        m = yy > 0
        if m.sum() < 10:
            continue
        ax.plot(xx[m], yy[m], linestyle="--", linewidth=1.0, alpha=0.55)
        ax.annotate(f"{C:.1e}", (xx[m][-1], yy[m][-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rollout FLOPs (search compute)")
    ax.set_ylabel("Update FLOPs (learning compute)")
    ax.set_title(f"Compute allocation plane (color = {color_col})")

    prettify(ax)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(color_col.replace("_", " "))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"allocation_map_{color_col}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, f"allocation_map_{color_col}.pdf"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------- Main ---------------------------
def main():
    set_pub_style()
    df = load_data()

    # Decide which metric to treat like "loss"
    # (you can swap gsm8k_accuracy for total_reward_mean if desired)
    metric_main = "gsm8k_accuracy"
    metric_reward = "total_reward_mean" if "total_reward_mean" in df.columns else None

    # 1) Approach 1
    plot_approach1_compute_scaling(df, ycol=metric_main)
    if metric_reward:
        plot_approach1_compute_scaling(df, ycol=metric_reward)

    # 2) Approach 2 IsoFLOPs (use your bands from earlier)
    # Adjust target_logC to match your observed buckets:
    # 16.05 (~1.1e16), 16.25 (~1.8e16), 16.45 (~2.8e16)
    plot_isoflop_curves(df, ycol=metric_main, target_logC=(16.05, 16.25, 16.45), tol=0.06)
    if metric_reward:
        plot_isoflop_curves(df, ycol=metric_reward, target_logC=(16.05, 16.25, 16.45), tol=0.06)

    # 3) Allocation map
    plot_allocation_map(df, color_col=metric_main)
    if metric_reward:
        plot_allocation_map(df, color_col=metric_reward)

    print(f"[Saved] figures to: {OUTDIR}/")

if __name__ == "__main__":
    main()
