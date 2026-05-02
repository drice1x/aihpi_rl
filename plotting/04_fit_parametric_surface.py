#!/usr/bin/env python3
# 04_fit_parametric_surface.py
#
# Fits a global parametric reward surface:
#   Rhat(Cu, Cr) = Rinf - A / (Cu^alpha) - B / (Cr^beta)
# using Huber loss, robust to outliers.
#
# Inputs: iso_alloc_plane_runs_total.csv
# Outputs: surface_fit_params.csv

import numpy as np
import pandas as pd
from dataclasses import dataclass

# SciPy is typically available; if not, you can switch to a simple grid search.
from scipy.optimize import minimize

INCSV = "iso_alloc_plane_runs_total.csv"
OUTCSV = "surface_fit_params.csv"
METRIC = "total_reward_mean"  # main objective
DELTA = 0.10                  # Huber delta in reward units (tune if needed)

EPS = 1e-12

@dataclass
class FitResult:
    Rinf: float
    A: float
    B: float
    alpha: float
    beta: float
    success: bool
    fun: float

def huber(resid, delta):
    a = np.abs(resid)
    quad = a <= delta
    out = np.empty_like(a)
    out[quad] = 0.5 * resid[quad]**2
    out[~quad] = delta * (a[~quad] - 0.5 * delta)
    return out

def unpack(theta):
    # unconstrained -> positive via softplus; exponents via softplus to keep >0
    # theta = [Rinf, logA, logB, logalpha, logbeta] in unconstrained space
    Rinf = theta[0]
    softplus = lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    A = softplus(theta[1]) + 1e-9
    B = softplus(theta[2]) + 1e-9
    alpha = softplus(theta[3]) + 1e-9
    beta  = softplus(theta[4]) + 1e-9
    return Rinf, A, B, alpha, beta

def predict(theta, Cu, Cr):
    Rinf, A, B, alpha, beta = unpack(theta)
    return Rinf - A / (np.maximum(Cu, EPS) ** alpha) - B / (np.maximum(Cr, EPS) ** beta)

def objective(theta, Cu, Cr, y):
    yhat = predict(theta, Cu, Cr)
    resid = yhat - y
    return float(np.mean(huber(resid, DELTA)))

def main():
    df = pd.read_csv(INCSV)

    for c in ["rollout_flops", "update_flops", METRIC]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["rollout_flops", "update_flops", METRIC])
    df = df[(df["rollout_flops"] > 0) & (df["update_flops"] > 0)]

    Cr = df["rollout_flops"].to_numpy(float)
    Cu = df["update_flops"].to_numpy(float)
    y  = df[METRIC].to_numpy(float)

    # Good initialization:
    # Rinf slightly above max reward (since model subtracts positive terms)
    Rinf0 = float(np.nanmax(y) + 0.2)
    theta0 = np.array([Rinf0, 1.0, 1.0, -2.0, -2.0], dtype=float)

    # Try multiple restarts (helps like Chinchilla)
    inits = [
        theta0,
        np.array([Rinf0, 0.1, 0.1, -1.0, -1.0]),
        np.array([Rinf0, 2.0, 2.0, -3.0, -3.0]),
        np.array([Rinf0, 1.0, 2.0, -2.0, -1.0]),
        np.array([Rinf0, 2.0, 1.0, -1.0, -2.0]),
    ]

    best = None
    for t0 in inits:
        res = minimize(
            objective, t0,
            args=(Cu, Cr, y),
            method="L-BFGS-B",
            options={"maxiter": 4000}
        )
        Rinf, A, B, alpha, beta = unpack(res.x)
        fr = FitResult(Rinf, A, B, alpha, beta, res.success, float(res.fun))
        if (best is None) or (fr.fun < best.fun):
            best = fr

    out = pd.DataFrame([{
        "metric": METRIC,
        "Rinf": best.Rinf,
        "A": best.A,
        "B": best.B,
        "alpha": best.alpha,
        "beta": best.beta,
        "huber_delta": DELTA,
        "objective": best.fun,
        "success": best.success,
        "n_points": len(df),
    }])
    out.to_csv(OUTCSV, index=False)
    print(f"[Saved] {OUTCSV}")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
