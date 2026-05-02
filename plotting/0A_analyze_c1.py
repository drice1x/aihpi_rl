#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_spot_C1.py

Plots for SPOT/NeurIPS DoE (new folder structure):
- compare reward types and rho cells
- allocation-centric plots (update_fraction, rollout vs update FLOPs)
- meant for merged CSV produced by analyze_merge_runs.py

Usage:
  python3 analyze_spot_C1.py \
    --csv all_runs_merged_C1.csv \
    --out figures_spot_C1 \
    --model_size 1.5 \
    --band C1

If your merged CSV contains only C1 already, you can omit --band.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

# ----------------------- Data ------------------------
NUM_COLS = [
    "model_size_B", "steps", "K", "L", "lora_r", "lora_alpha",
    "total_flops", "rollout_flops", "update_flops", "update_fraction",
    "gsm8k_accuracy",
    "reward_total_reward_mean",  # from flatten_reward_stats (if present)
    "reward_total_reward_std",
]

REQ_COLS = [
    "reward_mode", "rho_cell", "total_flops", "rollout_flops", "update_flops", "gsm8k_accuracy"
]

def load_data(path):
    df = pd.read_csv(path)

    # numeric coercions
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # compute total_flops if missing
    if "total_flops" in df.columns and "rollout_flops" in df.columns and "update_flops" in df.columns:
        bad = df["total_flops"].isna() | (df["total_flops"] <= 0)
        df.loc[bad, "total_flops"] = df.loc[bad, "rollout_flops"] + df.loc[bad, "update_flops"]

    # update_fraction if missing
    if "update_fraction" not in df.columns:
        if "update_flops" in df.columns and "total_flops" in df.columns:
            df["update_fraction"] = df["update_flops"] / df["total_flops"]

    # drop junk
    if "total_flops" in df.columns:
        df = df.dropna(subset=["total_flops"])
        df = df[df["total_flops"] > 0]

    return df

def apply_filters(df, band=None, model_size=None, lora_r=None, L=None, K=None):
    d = df.copy()

    if band and "band" in d.columns:
        d = d[d["band"].astype(str) == str(band)]

    if model_size is not None and "model_size_B" in d.columns:
        # tolerate float noise: keep within ±0.25B for 1.5B etc.
        ms = float(model_size)
        d = d[(d["model_size_B"] >= ms - 0.25) & (d["model_size_B"] <= ms + 0.25)]

    if lora_r is not None and "lora_r" in d.columns:
        d = d[d["lora_r"] == int(lora_r)]

    if L is not None and "L" in d.columns:
        d = d[d["L"] == int(L)]

    if K is not None and "K" in d.columns:
        d = d[d["K"] == int(K)]

    return d

def canonical_order(values, preferred):
    vals = [v for v in preferred if v in values]
    rest = sorted([v for v in values if v not in vals])
    return vals + rest


# ----------------------- Core plots ------------------------

def plot_accuracy_by_rho_and_reward(df, outdir):
    """
    Bar-like (point + errorbar) plot:
      x = rho_cell
      y = gsm8k_accuracy
      separate line per reward_mode
    """
    d = df.dropna(subset=["gsm8k_accuracy", "reward_mode", "rho_cell"]).copy()
    if d.empty:
        print("[WARN] No data for accuracy_by_rho_and_reward")
        return

    # aggregate across seeds (mean ± std)
    agg = (
        d.groupby(["reward_mode", "rho_cell"])["gsm8k_accuracy"]
         .agg(["mean", "std", "count"])
         .reset_index()
    )

    rewards = canonical_order(agg["reward_mode"].unique().tolist(),
                              ["sparse", "structured", "dense", "dense_verifier", "prm"])
    rhos = canonical_order(agg["rho_cell"].unique().tolist(),
                           ["rho15", "rho35", "rho50", "rho80"])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    x = np.arange(len(rhos))
    for rm in rewards:
        g = agg[agg["reward_mode"] == rm].set_index("rho_cell").reindex(rhos)
        y = g["mean"].to_numpy()
        e = g["std"].to_numpy()
        ax.plot(x, y, marker="o", linewidth=1.8, alpha=0.95, label=rm)
        # std error bars (only if count>1)
        ax.errorbar(x, y, yerr=e, fmt="none", capsize=2, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(rhos)
    ax.set_ylabel("GSM8K accuracy")
    ax.set_xlabel("rho cell (allocation policy)")
    ax.set_title("Accuracy vs rho cell (lines = reward mode; mean±std over seeds)")

    prettify(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "acc_vs_rho_by_reward.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "acc_vs_rho_by_reward.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_update_fraction_faceted(df, outdir):
    """
    Main allocation plot:
      x = update_fraction
      y = gsm8k_accuracy
      facet by reward_mode
      color/marker by rho_cell
    """
    d = df.dropna(subset=["gsm8k_accuracy", "update_fraction", "reward_mode", "rho_cell"]).copy()
    if d.empty:
        print("[WARN] No data for accuracy_vs_update_fraction_faceted")
        return

    rewards = canonical_order(d["reward_mode"].unique().tolist(),
                              ["sparse", "structured", "dense", "dense_verifier", "prm"])
    rhos = canonical_order(d["rho_cell"].unique().tolist(),
                           ["rho15", "rho35", "rho50", "rho80"])

    n = len(rewards)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.6, 3.6 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, rm in enumerate(rewards):
        ax = axes[i]
        g = d[d["reward_mode"] == rm].copy()
        for rho in rhos:
            h = g[g["rho_cell"] == rho]
            if h.empty:
                continue
            ax.scatter(h["update_fraction"], h["gsm8k_accuracy"], s=45, alpha=0.9, label=rho)

        ax.set_xlabel("Update fraction (update_flops / total_flops)")
        ax.set_ylabel("GSM8K accuracy")
        ax.set_title(f"Reward: {rm}")
        prettify(ax)
        ax.legend(frameon=False, loc="best")

    # hide unused axes
    for j in range(len(rewards), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "acc_vs_update_fraction_faceted.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "acc_vs_update_fraction_faceted.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_allocation_map_by_reward(df, outdir, color_col="gsm8k_accuracy"):
    """
    Allocation plane:
      x = rollout_flops
      y = update_flops
      color = accuracy
      facet by reward_mode
      annotate rho labels (optional)
    """
    need = ["rollout_flops", "update_flops", "total_flops", "reward_mode", color_col]
    d = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    d = d[(d["rollout_flops"] > 0) & (d["update_flops"] > 0)]
    if d.empty:
        print("[WARN] No data for allocation_map_by_reward")
        return

    rewards = canonical_order(d["reward_mode"].unique().tolist(),
                              ["sparse", "structured", "dense", "dense_verifier", "prm"])
    n = len(rewards)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.0, 4.0 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, rm in enumerate(rewards):
        ax = axes[i]
        g = d[d["reward_mode"] == rm].copy()

        sc = ax.scatter(
            g["rollout_flops"], g["update_flops"],
            c=g[color_col], s=55, alpha=0.9,
            edgecolors="black", linewidths=0.35
        )

        # Iso-total lines (choose a few budgets from that reward's points)
        xmin, xmax = g["rollout_flops"].min(), g["rollout_flops"].max()
        xx = np.logspace(np.log10(xmin), np.log10(xmax), 400)

        budgets = np.quantile(g["total_flops"], [0.25, 0.5, 0.75])
        for C in budgets:
            yy = C - xx
            m = yy > 0
            if m.sum() < 10:
                continue
            ax.plot(xx[m], yy[m], linestyle="--", linewidth=1.0, alpha=0.5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rollout FLOPs")
        ax.set_ylabel("Update FLOPs")
        ax.set_title(f"Allocation plane (reward={rm})")
        prettify(ax)

        # small rho annotations (only if not too crowded)
        if "rho_cell" in g.columns and len(g) <= 20:
            for _, r in g.iterrows():
                ax.annotate(str(r["rho_cell"]), (r["rollout_flops"], r["update_flops"]),
                            textcoords="offset points", xytext=(4, 2), fontsize=7, alpha=0.8)

        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color_col.replace("_", " "))

    for j in range(len(rewards), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"allocation_map_by_reward_{color_col}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"allocation_map_by_reward_{color_col}.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_total_flops_sanity(df, outdir):
    """
    Sanity: within a compute band, total_flops shouldn't vary wildly.
    Plot total_flops by rho and reward to see if 'band' is tight.
    """
    d = df.dropna(subset=["total_flops", "reward_mode", "rho_cell"]).copy()
    if d.empty:
        print("[WARN] No data for total_flops_sanity")
        return

    agg = (
        d.groupby(["reward_mode", "rho_cell"])["total_flops"]
         .agg(["mean", "std", "count"])
         .reset_index()
    )

    rewards = canonical_order(agg["reward_mode"].unique().tolist(),
                              ["sparse", "structured", "dense", "dense_verifier", "prm"])
    rhos = canonical_order(agg["rho_cell"].unique().tolist(),
                           ["rho15", "rho35", "rho50", "rho80"])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(rhos))

    for rm in rewards:
        g = agg[agg["reward_mode"] == rm].set_index("rho_cell").reindex(rhos)
        y = g["mean"].to_numpy()
        ax.plot(x, y, marker="o", linewidth=1.8, alpha=0.95, label=rm)

    ax.set_xticks(x)
    ax.set_xticklabels(rhos)
    ax.set_yscale("log")
    ax.set_ylabel("Total FLOPs (log scale)")
    ax.set_xlabel("rho cell")
    ax.set_title("Sanity: total FLOPs variation across rho and reward (should be banded)")
    prettify(ax)
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sanity_total_flops_by_rho_reward.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "sanity_total_flops_by_rho_reward.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------- Main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Merged CSV (e.g., all_runs_merged_C1.csv)")
    ap.add_argument("--out", type=str, default="figures_spot_C1")
    ap.add_argument("--band", type=str, default="", help="Optional band filter, e.g., C1")
    ap.add_argument("--model_size", type=float, default=1.5, help="Model size in B (default 1.5)")
    ap.add_argument("--lora_r", type=int, default=None, help="Optional LoRA rank filter")
    ap.add_argument("--L", type=int, default=None, help="Optional completion length filter")
    args = ap.parse_args()

    set_pub_style()
    ensure_dir(args.out)

    df = load_data(args.csv)

    # basic schema check
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        print("[WARN] Missing expected columns:", missing)
        print("[INFO] Available columns:", list(df.columns))
        # still continue; some plots may skip

    d = apply_filters(df, band=args.band or None, model_size=args.model_size, lora_r=args.lora_r, L=args.L)

    # main plots
    plot_total_flops_sanity(d, args.out)
    plot_accuracy_by_rho_and_reward(d, args.out)
    plot_accuracy_vs_update_fraction_faceted(d, args.out)
    plot_allocation_map_by_reward(d, args.out, color_col="gsm8k_accuracy")

    print(f"[Saved] figures to: {args.out}/")
    print(f"[Rows] {len(d)} (after filters)")

if __name__ == "__main__":
    main()