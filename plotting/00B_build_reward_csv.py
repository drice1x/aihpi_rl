#!/usr/bin/env python3
# 01_build_doe_csv_from_summary.py

import json
import os
import re
from pathlib import Path

import pandas as pd

ROOT = "runs_doe_C1"
OUTCSV = "iso_alloc_plane_runs_total.csv"


def parse_band_from_path(path_str: str):
    parts = Path(path_str).parts
    for p in parts:
        if re.fullmatch(r"C\d+", p):
            return p
    return None


def parse_rho_from_path(path_str: str):
    parts = Path(path_str).parts
    for p in parts:
        if re.fullmatch(r"rho\d+", p):
            return p
    return None


def parse_model_size_B(model_name: str):
    if not model_name:
        return None
    m = re.search(r"Qwen2\.5-(\d+(?:\.\d+)?)B", model_name)
    if m:
        return float(m.group(1))
    return None


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_rows(root: str):
    rows = []

    summary_paths = list(Path(root).rglob("logs/summary.json"))
    print(f"[INFO] Found {len(summary_paths)} summary.json files")

    for sp in summary_paths:
        try:
            with open(sp, "r") as f:
                js = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed reading {sp}: {e}")
            continue

        cfg = js.get("cfg", {})
        params = js.get("params", {})
        comp = js.get("compute_accounting", {})
        totals = comp.get("totals", {})
        rewards = js.get("rewards", {})

        output_dir = cfg.get("output_dir", str(sp.parent.parent))
        band = parse_band_from_path(output_dir)
        rho_cell = parse_rho_from_path(output_dir)

        reward_mode = cfg.get("reward_mode", None)
        model_name = cfg.get("model_name", None)
        model_size_B = parse_model_size_B(model_name)

        # Parse seed / K / L / S from output_dir as fallback if needed
        seed = cfg.get("seed", None)
        K = cfg.get("num_generations", None)
        L = cfg.get("max_completion_length", None)
        steps = cfg.get("max_steps", None)

        # Fallback parsing from path
        m_seed = re.search(r"/seed(\d+)(?:/|$)", output_dir)
        if seed is None and m_seed:
            seed = int(m_seed.group(1))

        m_kls = re.search(r"/K(\d+)_L(\d+)_S(\d+)(?:/|$)", output_dir)
        if m_kls:
            if K is None:
                K = int(m_kls.group(1))
            if L is None:
                L = int(m_kls.group(2))
            if steps is None:
                steps = int(m_kls.group(3))

        lora_r = cfg.get("lora_r", None)
        lora_alpha = cfg.get("lora_alpha", None)

        rollout_flops = totals.get("rollout_flops", None)
        update_flops = totals.get("update_flops", None)
        total_flops = totals.get("total_flops", None)
        rollout_tokens = totals.get("rollout_tokens", None)
        update_tokens = totals.get("update_tokens", None)

        if total_flops is None and rollout_flops is not None and update_flops is not None:
            total_flops = rollout_flops + update_flops

        update_fraction = None
        if total_flops not in (None, 0) and update_flops is not None:
            update_fraction = update_flops / total_flops

        row = {
            # identifiers
            "summary_path": str(sp),
            "output_dir": output_dir,
            "compute_band": band,
            "reward": reward_mode,
            "reward_mode": reward_mode,
            "rho_cell": rho_cell,

            # cfg
            "model_name": model_name,
            "model_size_B": model_size_B,
            "seed": seed,
            "steps": steps,
            "K": K,
            "L": L,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", None),
            "reuse_prefill_across_K": cfg.get("reuse_prefill_across_K", None),
            "dense_err_scale": cfg.get("dense_err_scale", None),
            "verifier_strict": cfg.get("verifier_strict", None),
            "verifier_friendly_prompt": cfg.get("verifier_friendly_prompt", None),
            "update_backbone_fraction": cfg.get("update_backbone_fraction", None),
            "prm_model_name": cfg.get("prm_model_name", None),
            "prm_device": cfg.get("prm_device", None),
            "prm_alpha": cfg.get("prm_alpha", None),
            "prm_max_steps_scored": cfg.get("prm_max_steps_scored", None),
            "prm_include_outcome": cfg.get("prm_include_outcome", None),
            "prm_outcome_scale": cfg.get("prm_outcome_scale", None),
            "reward_standardize": cfg.get("reward_standardize", None),
            "reward_clip_min": cfg.get("reward_clip_min", None),
            "reward_clip_max": cfg.get("reward_clip_max", None),
            "log_kl_proxy": cfg.get("log_kl_proxy", None),

            # params
            "total_params": params.get("total_params", None),
            "trainable_params_lora": params.get("trainable_params_lora", None),
            "backbone_params": params.get("backbone_params", None),
            "prm_params": params.get("prm_params", None),

            # compute accounting
            "N_params": comp.get("N_params", None),
            "R_params": comp.get("R_params", None),
            "c_forward": comp.get("c_forward", None),
            "c_backward": comp.get("c_backward", None),
            "f_lora": comp.get("f_lora", None),
            "update_backbone_fraction_a": comp.get("update_backbone_fraction_a", None),
            "update_effective_param_fraction": comp.get("update_effective_param_fraction", None),

            "rollout_tokens": rollout_tokens,
            "update_tokens": update_tokens,
            "rollout_flops": rollout_flops,
            "update_flops": update_flops,
            "total_flops": total_flops,
            "update_fraction": update_fraction,
        }

        # flatten reward summaries
        for reward_key, stats in rewards.items():
            if not isinstance(stats, dict):
                continue
            for stat_name, stat_val in stats.items():
                row[f"{reward_key}_{stat_name}"] = stat_val

        rows.append(row)

    return rows


def main():
    rows = build_rows(ROOT)
    if not rows:
        print("[WARN] No rows found.")
        return

    df = pd.DataFrame(rows)

    # Optional ordering of key columns
    preferred = [
        "summary_path", "output_dir", "compute_band",
        "reward", "reward_mode", "rho_cell",
        "model_name", "model_size_B", "seed",
        "steps", "K", "L", "lora_r", "lora_alpha",
        "rollout_tokens", "update_tokens",
        "rollout_flops", "update_flops", "total_flops", "update_fraction",
        "total_reward_mean", "total_reward_std", "total_reward_min", "total_reward_max",
    ]
    ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[ordered]

    df.to_csv(OUTCSV, index=False)

    print(f"[OK] Saved {len(df)} rows to {OUTCSV}")
    print("[INFO] Columns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()
