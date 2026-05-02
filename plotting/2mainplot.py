#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Inputs
ISO_MODEL = "isoCompute_model_results_mixed.csv"
ISO_K     = "isoK_results.csv"
ISO_L     = "isoL_results.csv"
ISO_LORA  = "isoLORA_results.csv"

# Output
OUTDIR = "figure"
OUTPDF = os.path.join(OUTDIR, "fig_main_multipanel_iso_compute2.pdf")
OUTPNG = os.path.join(OUTDIR, "fig_main_multipanel_iso_compute2.png")
os.makedirs(OUTDIR, exist_ok=True)

def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def prettify(ax):
    ax.grid(True, which="major", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def main():
    set_pub_style()

    dfM = pd.read_csv(ISO_MODEL)
    dfK = pd.read_csv(ISO_K)
    dfL = pd.read_csv(ISO_L)
    dfR = pd.read_csv(ISO_LORA) if os.path.exists(ISO_LORA) else None

    # numeric conversion
    for df in (dfM, dfK, dfL):
        for c in ["model_size_B","steps","K","L","total_flops","rollout_flops","update_flops",
                  "gsm8k_accuracy","n_eval_examples"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    if dfR is not None:
        for c in ["lora_r","steps","K","L","total_flops","rollout_flops","update_flops",
                  "gsm8k_accuracy","n_eval_examples","trainable_params_lora"]:
            if c in dfR.columns:
                dfR[c] = pd.to_numeric(dfR[c], errors="coerce")

    # ---- layout ----
    fig = plt.figure(figsize=(10.5, 7.5))
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.30)

    # ========== (A) Iso-compute model scaling ==========
    axA = fig.add_subplot(gs[0, 0])
    d = dfM.sort_values("model_size_B").copy()

    x = d["model_size_B"].values
    y = d["gsm8k_accuracy"].values
    axA.plot(x, y, marker="o")

    labels = d["label"].astype(str).values if "label" in d.columns else [f"{v:.1f}B" for v in x]
    for xi, yi, lab in zip(x, y, labels):
        axA.annotate(lab, (xi, yi), textcoords="offset points", xytext=(6, 4), ha="left")

    axA.set_xscale("log")
    axA.set_ylim(0.0, 0.75)
    axA.set_xlabel("Model size (B parameters, log scale)")
    axA.set_ylabel("GSM8K accuracy")
    axA.set_title("(A) Iso-compute post-training scaling")
    axA.text(0.03, 0.08, r"target $\approx 4.2\times 10^{15}$ FLOPs", transform=axA.transAxes)
    prettify(axA)

    # ========== (B) Iso-compute K sweep (NO ERROR BARS) ==========
    axB = fig.add_subplot(gs[0, 1])
    d = dfK.sort_values("K").copy()

    x = d["K"].values
    y = d["gsm8k_accuracy"].values
    axB.plot(x, y, marker="o")

    axB.set_ylim(0.15, 0.23)
    axB.set_xticks([2, 4, 8])
    axB.set_xlabel("Rollouts per prompt $K$")
    axB.set_ylabel("GSM8K accuracy")
    axB.set_title("(B) Iso-compute $K$ sweep (1.5B)")
    prettify(axB)

    # ========== (C) Iso-compute L sweep (NO ERROR BARS) ==========
    axC = fig.add_subplot(gs[1, 0])
    d = dfL.sort_values("L").copy()

    x = d["L"].values
    y = d["gsm8k_accuracy"].values
    axC.plot(x, y, marker="o")

    axC.set_ylim(0.15, 0.23)
    axC.set_xticks([512, 1024, 2048])
    axC.set_xlabel("Completion length cap $L$")
    axC.set_ylabel("GSM8K accuracy")
    axC.set_title("(C) Iso-compute length sweep (1.5B)")
    prettify(axC)

    # ========== (D) ONLY isoLoRA: learning compute vs accuracy ==========
    axD = fig.add_subplot(gs[1, 1])
    if dfR is None or len(dfR) == 0:
        axD.axis("off")
        axD.text(0.05, 0.6, "Missing isoLORA_results.csv", transform=axD.transAxes)
    else:
        d = dfR.sort_values("lora_r").copy()

        x = d["update_flops"].values
        y = d["gsm8k_accuracy"].values
        r = d["lora_r"].astype(int).values

        axD.plot(x, y, marker="^", linestyle="--", alpha=0.95)
        for xi, yi, ri in zip(x, y, r):
            axD.annotate(f"r={ri}", (xi, yi), textcoords="offset points", xytext=(6, 4), ha="left")

        axD.set_xscale("log")
        axD.set_ylim(0.15, 0.23)  # keep focus on the effect size you observe
        axD.set_xlabel("Learning compute (update FLOPs, log scale)")
        axD.set_ylabel("GSM8K accuracy")
        axD.set_title("(D) Learning-compute sweep (LoRA rank, 1.5B, iso-compute)")
        axD.text(0.03, 0.08, "fixed backbone (1.5B), fixed total FLOPs", transform=axD.transAxes)
        prettify(axD)

    fig.tight_layout(pad=0.6)
    fig.savefig(OUTPDF, bbox_inches="tight")
    fig.savefig(OUTPNG, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {OUTPDF}")
    print(f"[Saved] {OUTPNG}")

if __name__ == "__main__":
    main()
