#!/usr/bin/env python3
import numpy as np
import pandas as pd

INCSV = "iso_alloc_plane_runs_total.csv"
OUTCSV = "analysis_table2_with_validationruns.csv"

# Your three isoFLOP targets (log10)
TARGETS = [16.05, 16.25, 16.38]  # ~1.1e16, ~1.8e16, ~2.4e16
TOL = 0.06                       # band half-width in log10 space

def main():
    df = pd.read_csv(INCSV)

    # numeric hygiene
    for c in ["model_size_B","total_flops","rollout_flops","update_flops","update_fraction",
              "gsm8k_accuracy","total_reward_mean","K","L","steps","lora_r","lora_alpha"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # derive total_flops if missing
    if "total_flops" in df.columns and "rollout_flops" in df.columns and "update_flops" in df.columns:
        bad = df["total_flops"].isna() | (df["total_flops"] <= 0)
        df.loc[bad, "total_flops"] = df.loc[bad, "rollout_flops"] + df.loc[bad, "update_flops"]

    # derive update_fraction if missing
    if "update_fraction" not in df.columns and "update_flops" in df.columns and "total_flops" in df.columns:
        df["update_fraction"] = df["update_flops"] / df["total_flops"]

    df = df.dropna(subset=["total_flops","model_size_B"])
    df = df[(df["total_flops"] > 0) & (df["model_size_B"] > 0)]

    df["logC"] = np.log10(df["total_flops"])
    df["logN"] = np.log10(df["model_size_B"])
    # guard update_fraction
    if "update_fraction" in df.columns:
        df["rho"] = df["update_fraction"]
    else:
        df["rho"] = np.nan
    df["logrho"] = np.log10(df["rho"].where(df["rho"] > 0))

    # assign closest band (within tolerance)
    def assign_band(x):
        best = None
        bestdist = 1e9
        for t in TARGETS:
            d = abs(x - t)
            if d < bestdist:
                bestdist = d
                best = t
        if bestdist <= TOL:
            return best
        return np.nan

    df["band_logC"] = df["logC"].apply(assign_band)
    df = df.dropna(subset=["band_logC"])

    df.to_csv(OUTCSV, index=False)
    print(f"[Saved] {OUTCSV} (rows={len(df)})")
    print(df["band_logC"].value_counts().sort_index())

if __name__ == "__main__":
    main()
