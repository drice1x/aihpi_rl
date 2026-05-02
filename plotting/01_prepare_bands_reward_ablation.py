#!/usr/bin/env python3
import numpy as np
import pandas as pd

INCSV = "iso_alloc_plane_runs_reward_ablation.csv"
OUTCSV = "analysis_reward_ablation.csv"

TARGETS = [16.05, 16.25, 16.38]
TOL = 0.06

def main():
    df = pd.read_csv(INCSV)

    for c in ["model_size_B","total_flops","rollout_flops","update_flops","update_fraction",
              "total_reward_mean","gsm8k_accuracy","K","L","steps","lora_r"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "update_fraction" not in df.columns and "update_flops" in df.columns and "total_flops" in df.columns:
        df["update_fraction"] = df["update_flops"] / df["total_flops"]

    df = df.dropna(subset=["total_flops","update_fraction","reward_type"])
    df = df[(df["total_flops"] > 0) & (df["update_fraction"] > 0)]

    df["logC"] = np.log10(df["total_flops"])
    df["rho"] = df["update_fraction"]
    df["logrho"] = np.log10(df["rho"])

    def assign_band(x):
        best, bestdist = None, 1e9
        for t in TARGETS:
            d = abs(x - t)
            if d < bestdist:
                bestdist = d
                best = t
        return best if bestdist <= TOL else np.nan

    df["band_logC"] = df["logC"].apply(assign_band)
    df = df.dropna(subset=["band_logC"])

    df.to_csv(OUTCSV, index=False)
    print(f"[Saved] {OUTCSV} rows={len(df)}")
    print(df.groupby(["reward_type","band_logC"]).size())

if __name__ == "__main__":
    main()
