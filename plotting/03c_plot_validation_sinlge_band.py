#!/usr/bin/env python3
# 03c_plot_validation_single_band.py
#
# Figure 4: Predictive validation beyond the sweep grid for ONE compute band.
# Fits the allocation curve using only "sweep" points, then overlays "validation" points.
#
# Heuristic by default:
#   validation if lora_r >= 192 OR run name contains prefill/reuse/val/extra
#
# Adjust BAND_TARGET_LOGC and TOL to match your experiment narrative.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = "analysis_table2_with_validationruns.csv"  # from 01_prepare_bands.py
OUTDIR = "figures_spot_allocation_fit2"
os.makedirs(OUTDIR, exist_ok=True)

# --- Figure 4 config ---
METRIC = "total_reward_mean"    # main claim should use reward; switch if needed
BAND_TARGET_LOGC = 16.25        # pick the band you validated on
TOL = 1e-9                      # since you already banded, you can keep exact; set e.g. 0.03 if needed

# Validation identification (customize to your runs)
VALID_LORA_R_MIN = 192
VALID_RUN_SUBSTRINGS = ["prefill", "reuse", "val", "extra", "validation"]

# Fit model: quadratic in logrho
RHO_GRID = np.logspace(np.log10(0.005), np.log10(0.30), 600)

def prettify(ax):
    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", alpha=0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def is_validation_row(row):
    r = row.get("lora_r", np.nan)
    name = str(row.get("run", "")).lower()
    if np.isfinite(r) and r >= VALID_LORA_R_MIN:
        return True
    for s in VALID_RUN_SUBSTRINGS:
        if s in name:
            return True
    return False

def fit_quad_logrho(x_logrho, y):
    x = np.asarray(x_logrho, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(y) < 5:
        return None
    a, b, c = np.polyfit(x, y, deg=2)
    return float(a), float(b), float(c)

def main():
    df = pd.read_csv(DATA)

    # numeric coercion
    for c in ["band_logC", "rho", "update_fraction", "logrho", "lora_r",
              "total_flops", "rollout_flops", "update_flops", METRIC]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "rho" not in df.columns:
        if "update_fraction" in df.columns:
            df["rho"] = df["update_fraction"]
        else:
            raise ValueError("Need rho or update_fraction in analysis_table2_with_validationruns.csv")

    if "logrho" not in df.columns:
        df["logrho"] = np.log10(df["rho"].where(df["rho"] > 0))

    # Select band
    band_sel = df.dropna(subset=["band_logC"]).copy()
    band_sel = band_sel[np.abs(band_sel["band_logC"] - BAND_TARGET_LOGC) <= TOL].copy()
    if len(band_sel) == 0:
        # fallback: exact match if already banded
        band_sel = df[df["band_logC"] == BAND_TARGET_LOGC].copy()

    band_sel = band_sel.dropna(subset=["rho", "logrho", METRIC])
    if len(band_sel) < 5:
        raise RuntimeError(f"Not enough points in band {BAND_TARGET_LOGC} for metric {METRIC}.")

    # Tag sweep vs validation
    band_sel["is_validation"] = band_sel.apply(is_validation_row, axis=1)

    sweep = band_sel[~band_sel["is_validation"]].copy()
    valid = band_sel[band_sel["is_validation"]].copy()

    if len(sweep) < 5:
        raise RuntimeError("Not enough sweep (non-validation) points to fit. "
                           "Adjust validation heuristic or add more sweep points.")

    # Fit only on sweep points
    coeff = fit_quad_logrho(sweep["logrho"].to_numpy(), sweep[METRIC].to_numpy())
    if coeff is None:
        raise RuntimeError("Fit failed (need >=5 sweep points with finite logrho and metric).")
    a, b, c = coeff

    logr_grid = np.log10(RHO_GRID)
    yhat = a * logr_grid**2 + b * logr_grid + c
    j = int(np.nanargmax(yhat))
    rho_star = float(RHO_GRID[j])
    y_star = float(yhat[j])

    # Best sweep and best validation (maximize metric; for reward, higher is better even if negative)
    best_sweep = sweep.loc[sweep[METRIC].idxmax()]
    best_valid = valid.loc[valid[METRIC].idxmax()] if len(valid) > 0 else None

    # Sweep rho range (for "clipped optimum" visuals)
    rho_min = float(sweep["rho"].min())
    rho_max = float(sweep["rho"].max())

    # Plot
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    # scatter sweep + validation
    ax.scatter(sweep["rho"], sweep[METRIC], s=60, alpha=0.9, label="Sweep (fit data)")
    if len(valid) > 0:
        ax.scatter(valid["rho"], valid[METRIC], s=95, marker="^", alpha=0.95, label="Validation (out-of-grid)")

    # fitted curve
    ax.plot(RHO_GRID, yhat, linewidth=1.7, alpha=0.95, label="Fit (quadratic in log ρ)")

    # predicted optimum marker
    ax.scatter([rho_star], [y_star], s=160, marker="X", edgecolors="black", linewidths=0.7, label=r"Predicted optimum $\rho^*$")

    # mark best points
    ax.scatter([best_sweep["rho"]], [best_sweep[METRIC]], s=120, marker="o", edgecolors="black", linewidths=0.6,
               label="Best in sweep")
    if best_valid is not None:
        ax.scatter([best_valid["rho"]], [best_valid[METRIC]], s=140, marker="^", edgecolors="black", linewidths=0.6,
                   label="Best validation")

    # show sweep explored range (for “clipped” story)
    ax.axvline(rho_min, linestyle=":", linewidth=1.2, alpha=0.7)
    ax.axvline(rho_max, linestyle=":", linewidth=1.2, alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel(r"Update fraction $\rho = \mathrm{FLOPs}_{update}/\mathrm{FLOPs}_{total}$")
    ax.set_ylabel(METRIC.replace("_", " "))
    ax.set_title(f"Validation beyond sweep grid (log10(C)≈{BAND_TARGET_LOGC:.2f})")

    prettify(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    out_base = os.path.join(OUTDIR, f"validation_band{BAND_TARGET_LOGC:.2f}_{METRIC}".replace(".", "p"))
    fig.savefig(out_base + ".png", bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] Figure 4 to {out_base}.(png/pdf)")
    print(f"  Fit points: {len(sweep)} sweep, {len(valid)} validation")
    print(f"  Predicted rho*: {rho_star:.4f}, predicted metric: {y_star:.4f}")
    print(f"  Best sweep: rho={best_sweep['rho']:.4f}, {METRIC}={best_sweep[METRIC]:.4f}, run={best_sweep.get('run','')}")
    if best_valid is not None:
        print(f"  Best val  : rho={best_valid['rho']:.4f}, {METRIC}={best_valid[METRIC]:.4f}, run={best_valid.get('run','')}")

if __name__ == "__main__":
    main()
