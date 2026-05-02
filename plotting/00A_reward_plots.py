#!/usr/bin/env python3
# paper_ready_reward_plots.py
#
# Paper-ready plots for reward-structure-aware GRPO scaling.
#
# Generates:
#   1) Faceted compute allocation planes by reward type (non-PRM main figure)
#   2) Optional separate PRM allocation plane
#   3) IsoFLOP curves: performance vs achieved update fraction rho, one line per reward
#   4) Best-achieved frontier: performance vs total compute, one curve per reward
#
# Assumptions:
#   - CSV has at least:
#       rollout_flops, update_flops, total_flops, gsm8k_accuracy
#   - Reward column can be one of:
#       reward_mode, reward, reward_type
#   - Optional:
#       model_size_B, update_fraction
#
# Usage:
#   python paper_ready_reward_plots.py \
#       --incsv iso_alloc_plane_runs_total.csv \
#       --outdir figures_reward_paper \
#       --metric gsm8k_accuracy
#
# Notes:
#   - Cross-reward comparisons should use accuracy, not raw reward.
#   - PRM is plotted separately by default because it includes auxiliary reward-model compute.

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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

# ----------------------- Helpers -----------------------

def detect_reward_col(df: pd.DataFrame) -> str:
    for c in ["reward_mode", "reward", "reward_type"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find reward column. Expected one of: reward_mode, reward, reward_type")

def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def bucket_model_size(ms):
    if pd.isna(ms):
        return "unknown"
    ms = float(ms)
    if ms < 1.0:
        return "0.5B"
    if ms < 2.2:
        return "1.5B"
    if ms < 5.0:
        return "3B"
    return "7B"

def compute_axis_limits(df, xcol, ycol, pad_log=0.08):
    xv = df[xcol].to_numpy(dtype=float)
    yv = df[ycol].to_numpy(dtype=float)
    xv = xv[np.isfinite(xv) & (xv > 0)]
    yv = yv[np.isfinite(yv) & (yv > 0)]
    lx0, lx1 = np.log10(xv.min()), np.log10(xv.max())
    ly0, ly1 = np.log10(yv.min()), np.log10(yv.max())
    return (10 ** (lx0 - pad_log), 10 ** (lx1 + pad_log),
            10 ** (ly0 - pad_log), 10 ** (ly1 + pad_log))

def add_isototal_lines(ax, df_subset, n_lines=4, color="gray", alpha=0.45):
    d = df_subset.dropna(subset=["rollout_flops", "update_flops", "total_flops"]).copy()
    d = d[(d["rollout_flops"] > 0) & (d["update_flops"] > 0) & (d["total_flops"] > 0)]
    if len(d) < 4:
        return

    xmin = d["rollout_flops"].min()
    xmax = d["rollout_flops"].max()
    xx = np.logspace(np.log10(xmin), np.log10(xmax), 400)

    budgets = np.quantile(d["total_flops"], np.linspace(0.15, 0.85, n_lines))
    budgets = np.unique(budgets)

    for C in budgets:
        yy = C - xx
        m = yy > 0
        if m.sum() < 10:
            continue
        ax.plot(xx[m], yy[m], linestyle="--", linewidth=1.0, alpha=alpha, color=color)

def add_model_markers(ax, g, metric, cmap, norm):
    marker_map = {
        "0.5B": "o",
        "1.5B": "s",
        "3B": "^",
        "7B": "D",
        "unknown": "o",
    }
    if "model_size_B" in g.columns:
        g = g.copy()
        g["size_bucket"] = g["model_size_B"].apply(bucket_model_size)
    else:
        g = g.copy()
        g["size_bucket"] = "unknown"

    for bucket, gb in g.groupby("size_bucket"):
        ax.scatter(
            gb["rollout_flops"],
            gb["update_flops"],
            c=gb[metric],
            cmap=cmap,
            norm=norm,
            s=56,
            alpha=0.92,
            edgecolors="black",
            linewidths=0.35,
            marker=marker_map.get(bucket, "o"),
            label=bucket,
        )

# ----------------------- Data ------------------------

def load_data(incsv: str) -> pd.DataFrame:
    df = pd.read_csv(incsv)

    num_cols = [
        "model_size_B", "steps", "K", "L", "lora_r", "lora_alpha",
        "total_flops", "rollout_flops", "update_flops", "update_fraction",
        "gsm8k_accuracy", "total_reward_mean", "total_reward_std",
    ]
    ensure_numeric(df, num_cols)

    if "total_flops" in df.columns and "rollout_flops" in df.columns and "update_flops" in df.columns:
        bad = df["total_flops"].isna() | (df["total_flops"] <= 0)
        df.loc[bad, "total_flops"] = df.loc[bad, "rollout_flops"] + df.loc[bad, "update_flops"]

    if "update_fraction" not in df.columns and {"update_flops", "total_flops"}.issubset(df.columns):
        df["update_fraction"] = df["update_flops"] / df["total_flops"]

    reward_col = detect_reward_col(df)
    df[reward_col] = df[reward_col].astype(str).str.strip()

    keep = ["rollout_flops", "update_flops", "total_flops"]
    df = df.dropna(subset=[c for c in keep if c in df.columns]).copy()
    df = df[(df["rollout_flops"] > 0) & (df["update_flops"] > 0) & (df["total_flops"] > 0)]
    return df

# ------------------- Plot 1: Faceted allocation planes ----------------------

def plot_faceted_allocation_planes(
    df: pd.DataFrame,
    outdir: str,
    metric: str = "gsm8k_accuracy",
    exclude_prm: bool = True,
):
    reward_col = detect_reward_col(df)

    d = df.dropna(subset=["rollout_flops", "update_flops", "total_flops", metric]).copy()
    if exclude_prm:
        d = d[d[reward_col].str.lower() != "prm"].copy()

    rewards = [r for r in ["sparse", "structured", "dense", "dense_verifier"] if r in set(d[reward_col])]
    if len(rewards) == 0:
        print("[WARN] No non-PRM rewards found for faceted allocation planes.")
        return

    x0, x1, y0, y1 = compute_axis_limits(d, "rollout_flops", "update_flops")
    vmin, vmax = np.nanmin(d[metric].to_numpy()), np.nanmax(d[metric].to_numpy())
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.0), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, reward in zip(axes, rewards):
        g = d[d[reward_col] == reward].copy()
        if len(g) == 0:
            ax.set_visible(False)
            continue

        add_isototal_lines(ax, g, n_lines=4)
        add_model_markers(ax, g, metric, cmap, norm)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(reward.replace("_", " "))
        prettify(ax)

    for k in range(len(rewards), 4):
        axes[k].set_visible(False)

    for ax in axes[2:]:
        if ax.get_visible():
            ax.set_xlabel("Rollout FLOPs (search compute)")
    for ax in [axes[0], axes[2]]:
        if ax.get_visible():
            ax.set_ylabel("Update FLOPs (learning compute)")

    # shared colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.035, pad=0.02)
    cbar.set_label(metric.replace("_", " "))

    # marker legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=7, label="0.5B"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=7, label="1.5B"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=7, label="3B"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=7, label="7B"),
    ]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Compute allocation plane by reward structure (non-PRM)", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    png = os.path.join(outdir, f"facet_allocation_plane_nonprm_{metric}.png")
    pdf = os.path.join(outdir, f"facet_allocation_plane_nonprm_{metric}.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

def plot_prm_allocation_plane(df: pd.DataFrame, outdir: str, metric: str = "gsm8k_accuracy"):
    reward_col = detect_reward_col(df)
    d = df.dropna(subset=["rollout_flops", "update_flops", "total_flops", metric]).copy()
    d = d[d[reward_col].str.lower() == "prm"].copy()

    if len(d) < 3:
        print("[WARN] Not enough PRM points for separate allocation plane.")
        return

    vmin, vmax = np.nanmin(df[metric].to_numpy()), np.nanmax(df[metric].to_numpy())
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    add_isototal_lines(ax, d, n_lines=4)
    add_model_markers(ax, d, metric, cmap, norm)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rollout FLOPs (search compute)")
    ax.set_ylabel("Update FLOPs (learning compute)")
    ax.set_title("PRM allocation plane")
    prettify(ax)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(metric.replace("_", " "))

    fig.tight_layout()
    png = os.path.join(outdir, f"allocation_plane_prm_{metric}.png")
    pdf = os.path.join(outdir, f"allocation_plane_prm_{metric}.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

# ------------------- Plot 2: IsoFLOP rho curves by reward ----------------------

def plot_isoflop_rho_curves_by_reward(
    df: pd.DataFrame,
    outdir: str,
    metric: str = "gsm8k_accuracy",
    target_logC=(16.05, 16.25, 16.45),
    tol=0.06,
    include_prm=False,
):
    reward_col = detect_reward_col(df)
    d = df.dropna(subset=["total_flops", "update_fraction", metric]).copy()
    d["logC"] = log10_safe(d["total_flops"].to_numpy())

    if not include_prm:
        d = d[d[reward_col].str.lower() != "prm"].copy()

    rewards = [r for r in ["sparse", "structured", "dense", "dense_verifier", "prm"] if r in set(d[reward_col])]
    if len(rewards) == 0:
        print("[WARN] No rewards available for IsoFLOP rho curves.")
        return

    n_panels = len(target_logC)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.8), sharey=True)
    if n_panels == 1:
        axes = [axes]

    color_map = {
        "sparse": "#2ca02c",
        "structured": "#d62728",
        "dense": "#1f77b4",
        "dense_verifier": "#ff7f0e",
        "prm": "#9467bd",
    }

    for ax, t in zip(axes, target_logC):
        band = d[(d["logC"] >= t - tol) & (d["logC"] <= t + tol)].copy()
        if len(band) == 0:
            ax.set_visible(False)
            continue

        for reward in rewards:
            g = band[band[reward_col] == reward].copy()
            if len(g) < 2:
                continue

            g = g.sort_values("update_fraction")
            x = g["update_fraction"].to_numpy()
            y = g[metric].to_numpy()

            ax.scatter(x, y, s=34, alpha=0.9, color=color_map.get(reward, None), label=reward)
            ax.plot(x, y, linewidth=1.5, alpha=0.9, color=color_map.get(reward, None))

        ax.set_xscale("log")
        ax.set_xlabel("Achieved update fraction $\\rho$")
        ax.set_title(f"log10(C) ≈ {t:.2f}")
        prettify(ax)

    axes[0].set_ylabel(metric.replace("_", " "))
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=min(5, len(labels)), bbox_to_anchor=(0.5, 1.02))

    title_suffix = "including PRM" if include_prm else "non-PRM"
    fig.suptitle(f"IsoFLOP curves: performance vs achieved update fraction ({title_suffix})", y=1.06, fontsize=12)
    fig.tight_layout()

    name = f"isoflop_rho_curves_by_reward_{metric}_{'withprm' if include_prm else 'nonprm'}"
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)

# ------------------- Plot 3: Optimal rho* vs compute band ----------------------

def fit_optimal_rho(g: pd.DataFrame, metric: str):
    g = g.dropna(subset=["update_fraction", metric]).copy()
    g = g[(g["update_fraction"] > 0)]
    if len(g) < 4:
        return np.nan

    x = np.log10(g["update_fraction"].to_numpy(dtype=float))
    y = g[metric].to_numpy(dtype=float)

    if len(np.unique(np.round(x, 4))) < 3:
        idx = np.nanargmax(y)
        return float(g.iloc[idx]["update_fraction"])

    coeff = np.polyfit(x, y, deg=2)  # y = ax^2 + bx + c
    a, b, _ = coeff
    if abs(a) < 1e-12:
        idx = np.nanargmax(y)
        return float(g.iloc[idx]["update_fraction"])

    x_star = -b / (2 * a)

    # If parabola opens up or optimum falls outside, fallback to empirical best
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    if (a >= 0) or (x_star < x_min) or (x_star > x_max):
        idx = np.nanargmax(y)
        return float(g.iloc[idx]["update_fraction"])

    return float(10 ** x_star)

def plot_optimal_rho_vs_compute(
    df: pd.DataFrame,
    outdir: str,
    metric: str = "gsm8k_accuracy",
    target_logC=(16.05, 16.25, 16.45),
    tol=0.06,
    include_prm=False,
):
    reward_col = detect_reward_col(df)
    d = df.dropna(subset=["total_flops", "update_fraction", metric]).copy()
    d["logC"] = log10_safe(d["total_flops"].to_numpy())

    if not include_prm:
        d = d[d[reward_col].str.lower() != "prm"].copy()

    rewards = [r for r in ["sparse", "structured", "dense", "dense_verifier", "prm"] if r in set(d[reward_col])]
    if len(rewards) == 0:
        print("[WARN] No rewards available for rho* plot.")
        return

    rows = []
    for reward in rewards:
        for t in target_logC:
            band = d[(d["logC"] >= t - tol) & (d["logC"] <= t + tol) & (d[reward_col] == reward)].copy()
            if len(band) < 3:
                continue
            rho_star = fit_optimal_rho(band, metric)
            rows.append({"reward": reward, "logC_target": t, "rho_star": rho_star})

    if len(rows) == 0:
        print("[WARN] Could not estimate rho* for any reward/band.")
        return

    rhos = pd.DataFrame(rows)
    color_map = {
        "sparse": "#2ca02c",
        "structured": "#d62728",
        "dense": "#1f77b4",
        "dense_verifier": "#ff7f0e",
        "prm": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for reward, g in rhos.groupby("reward"):
        g = g.sort_values("logC_target")
        ax.plot(10 ** g["logC_target"], g["rho_star"], marker="o", linewidth=1.8, label=reward, color=color_map.get(reward, None))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Target compute band center")
    ax.set_ylabel("Estimated optimal update fraction $\\rho^*$")
    ax.set_title(f"Compute-optimal update allocation by reward type")
    prettify(ax)
    ax.legend(frameon=False)

    fig.tight_layout()
    name = f"optimal_rho_vs_compute_{metric}_{'withprm' if include_prm else 'nonprm'}"
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)

# ------------------- Plot 4: Frontier by reward ----------------------

def frontier_upper_envelope(g: pd.DataFrame, metric: str, n_bins=14):
    g = g.dropna(subset=["total_flops", metric]).copy()
    if len(g) < 3:
        return None, None
    logC = log10_safe(g["total_flops"].to_numpy())
    bins = np.linspace(np.nanmin(logC), np.nanmax(logC), n_bins)
    idx = np.digitize(logC, bins)

    env_x, env_y = [], []
    for k in range(1, len(bins) + 1):
        sel = g[idx == k]
        if len(sel) == 0:
            continue
        j = sel[metric].idxmax()
        env_x.append(g.loc[j, "total_flops"])
        env_y.append(g.loc[j, metric])
    return env_x, env_y

def plot_frontier_by_reward(
    df: pd.DataFrame,
    outdir: str,
    metric: str = "gsm8k_accuracy",
    include_prm=False,
):
    reward_col = detect_reward_col(df)
    d = df.dropna(subset=["total_flops", metric]).copy()
    if not include_prm:
        d = d[d[reward_col].str.lower() != "prm"].copy()

    rewards = [r for r in ["sparse", "structured", "dense", "dense_verifier", "prm"] if r in set(d[reward_col])]
    if len(rewards) == 0:
        print("[WARN] No rewards available for frontier plot.")
        return

    color_map = {
        "sparse": "#2ca02c",
        "structured": "#d62728",
        "dense": "#1f77b4",
        "dense_verifier": "#ff7f0e",
        "prm": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for reward in rewards:
        g = d[d[reward_col] == reward].copy()
        ax.scatter(g["total_flops"], g[metric], s=22, alpha=0.35, color=color_map.get(reward, None))
        env_x, env_y = frontier_upper_envelope(g, metric)
        if env_x is not None and len(env_x) >= 2:
            ax.plot(env_x, env_y, linewidth=2.0, label=reward, color=color_map.get(reward, None))

    ax.set_xscale("log")
    ax.set_xlabel("Total post-training FLOPs")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Best-achieved compute frontier by reward type")
    prettify(ax)
    ax.legend(frameon=False)

    fig.tight_layout()
    name = f"frontier_by_reward_{metric}_{'withprm' if include_prm else 'nonprm'}"
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incsv", type=str, default="iso_alloc_plane_runs_total.csv")
    ap.add_argument("--outdir", type=str, default="figures_reward_paper")
    ap.add_argument("--metric", type=str, default="gsm8k_accuracy")
    ap.add_argument("--target_logC", type=float, nargs="+", default=[16.05, 16.25, 16.45])
    ap.add_argument("--tol", type=float, default=0.06)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_pub_style()

    df = load_data(args.incsv)

    if args.metric not in df.columns:
        raise ValueError(f"Metric column '{args.metric}' not found in CSV. Available columns: {list(df.columns)}")

    # Main paper plots
    plot_faceted_allocation_planes(df, args.outdir, metric=args.metric, exclude_prm=True)
    plot_isoflop_rho_curves_by_reward(df, args.outdir, metric=args.metric, target_logC=tuple(args.target_logC), tol=args.tol, include_prm=False)
    plot_optimal_rho_vs_compute(df, args.outdir, metric=args.metric, target_logC=tuple(args.target_logC), tol=args.tol, include_prm=False)
    plot_frontier_by_reward(df, args.outdir, metric=args.metric, include_prm=False)

    # Separate PRM figure(s)
    plot_prm_allocation_plane(df, args.outdir, metric=args.metric)
    plot_isoflop_rho_curves_by_reward(df, args.outdir, metric=args.metric, target_logC=tuple(args.target_logC), tol=args.tol, include_prm=True)
    plot_optimal_rho_vs_compute(df, args.outdir, metric=args.metric, target_logC=tuple(args.target_logC), tol=args.tol, include_prm=True)
    plot_frontier_by_reward(df, args.outdir, metric=args.metric, include_prm=True)

    print(f"[Saved] figures to: {args.outdir}/")

if __name__ == "__main__":
    main()
