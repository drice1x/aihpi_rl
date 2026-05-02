#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List


STAGE_ROOTS = {
    "stageA": "runs_spot_alloc_scaling",
    "stageB": "runs_spot_reward_extension",
    "stageC": "runs_spot_tpprm_judge_scaling",
    "stageD": "runs_spot_real_tpprm",
}

REWARD_MODES = {
    "sparse",
    "structured",
    "dense",
    "dense_verifier",
    "prm",
    "prm_scalar",
    "prm_tp",
}

JUDGE_TAGS = {"j15", "j3", "j7", "j14", "prm7", "prm72"}


def safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def to_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def log10_or_none(x):
    x = to_float(x)
    if x is None or x <= 0:
        return None
    return math.log10(x)


def extract_path_metadata(summary_path: Path, stage: str) -> Dict[str, Any]:
    parts = list(summary_path.parts)

    meta: Dict[str, Any] = {
        "stage": stage,
        "scope": None,
        "compute_band": None,
        "reward_path": None,
        "judge_tag": None,
        "alloc": None,
        "model_tag": None,
        "judge_model_path": None,
        "lora_rank_path": None,
        "seed_path": None,
        "K_path": None,
        "L_path": None,
        "S_path": None,
    }

    for i, part in enumerate(parts):
        if re.fullmatch(r"C\d+", part):
            meta["compute_band"] = part
            if i > 0:
                meta["scope"] = parts[i - 1]

        elif part in REWARD_MODES:
            meta["reward_path"] = part

        elif part in JUDGE_TAGS:
            meta["judge_tag"] = part

        elif re.fullmatch(r"a\d+", part):
            meta["alloc"] = part

        elif re.fullmatch(r"r\d+", part):
            meta["lora_rank_path"] = int(part[1:])

        elif re.fullmatch(r"seed\d+", part):
            meta["seed_path"] = int(part[4:])

        elif re.fullmatch(r"K\d+_L\d+_S\d+", part):
            m = re.fullmatch(r"K(\d+)_L(\d+)_S(\d+)", part)
            if m:
                meta["K_path"] = int(m.group(1))
                meta["L_path"] = int(m.group(2))
                meta["S_path"] = int(m.group(3))

        elif "__" in part or part.startswith("Qwen__"):
            if meta["model_tag"] is None:
                meta["model_tag"] = part
            else:
                meta["judge_model_path"] = part

    return meta


def reward_mean(rewards: Dict[str, Any], name: str):
    return safe_get(rewards, name, "mean")


def reward_std(rewards: Dict[str, Any], name: str):
    return safe_get(rewards, name, "std")


def choose_correctness_mean(reward_mode: str | None, rewards: Dict[str, Any]):
    if reward_mode == "sparse":
        return reward_mean(rewards, "sparse_correct")
    if reward_mode == "dense":
        return reward_mean(rewards, "dense_correct")
    if reward_mode == "dense_verifier":
        return reward_mean(rewards, "verifier_correct")
    if reward_mode == "prm_scalar":
        return reward_mean(rewards, "prm_correct")
    if reward_mode == "prm_tp":
        return reward_mean(rewards, "prm_tp_correct")
    if reward_mode == "structured":
        # In your structured reward, check_answer is a shaped score, not binary.
        # Prefer validation accuracy if available for true correctness.
        return reward_mean(rewards, "check_answer")
    return None


def flatten_summary(summary: Dict[str, Any], summary_path: Path, stage: str) -> Dict[str, Any]:
    cfg = safe_get(summary, "cfg", default={}) or {}
    params = safe_get(summary, "params", default={}) or {}
    comp = safe_get(summary, "compute_accounting", default={}) or {}
    totals = safe_get(comp, "totals", default={}) or {}
    rewards = safe_get(summary, "rewards", default={}) or {}
    validation = safe_get(summary, "validation", default={}) or {}

    row: Dict[str, Any] = {
        "stage": stage,
        "summary_path": str(summary_path),
        "run_dir": str(summary_path.parent.parent if summary_path.parent.name == "logs" else summary_path.parent),
    }

    row.update(extract_path_metadata(summary_path, stage))

    row.update({
        # Core config
        "model_name": cfg.get("model_name"),
        "output_dir": cfg.get("output_dir"),
        "dataset_name": cfg.get("dataset_name"),
        "reward_mode": cfg.get("reward_mode"),
        "seed": cfg.get("seed"),

        # Training setup
        "max_steps": cfg.get("max_steps"),
        "num_generations": cfg.get("num_generations"),
        "max_prompt_length": cfg.get("max_prompt_length"),
        "max_completion_length": cfg.get("max_completion_length"),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps"),
        "per_device_train_batch_size": cfg.get("per_device_train_batch_size"),
        "learning_rate": cfg.get("learning_rate"),
        "weight_decay": cfg.get("weight_decay"),
        "warmup_ratio": cfg.get("warmup_ratio"),
        "lr_scheduler_type": cfg.get("lr_scheduler_type"),
        "optim": cfg.get("optim"),
        "temperature": cfg.get("temperature"),
        "kl_beta": cfg.get("kl_beta"),
        "reuse_prefill_across_K": cfg.get("reuse_prefill_across_K"),

        # LoRA
        "lora_r": cfg.get("lora_r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "lora_dropout": cfg.get("lora_dropout"),
        "use_peft": cfg.get("use_peft"),
        "full_finetuning": cfg.get("full_finetuning"),

        # Compute accounting
        "update_backbone_fraction": cfg.get("update_backbone_fraction"),
        "flops_scale_forward": cfg.get("flops_scale_forward"),
        "flops_scale_backward": cfg.get("flops_scale_backward"),
        "cfg_reward_model_params": cfg.get("reward_model_params"),

        # Reward config
        "dense_err_scale": cfg.get("dense_err_scale"),
        "reward_standardize": cfg.get("reward_standardize"),
        "reward_clip_min": cfg.get("reward_clip_min"),
        "reward_clip_max": cfg.get("reward_clip_max"),

        # PRM scalar / real PRM
        "prm_model_name": cfg.get("prm_model_name"),
        "prm_device": cfg.get("prm_device"),
        "prm_alpha": cfg.get("prm_alpha"),
        "prm_max_steps_scored": cfg.get("prm_max_steps_scored"),
        "prm_include_outcome": cfg.get("prm_include_outcome"),
        "prm_outcome_scale": cfg.get("prm_outcome_scale"),

        # Stage C / TP-PRM verifier fields
        "verifier_model_name": cfg.get("verifier_model_name"),
        "verifier_device": cfg.get("verifier_device"),
        "verifier_max_new_tokens": cfg.get("verifier_max_new_tokens"),
        "thought_sep_mode": cfg.get("thought_sep_mode"),
        "tp_merge_same_sign": cfg.get("tp_merge_same_sign"),
        "tp_capability_adaptive": cfg.get("tp_capability_adaptive"),
        "tp_correct_thought_reward": cfg.get("tp_correct_thought_reward"),
        "tp_incorrect_thought_penalty": cfg.get("tp_incorrect_thought_penalty"),
        "tp_neutral_thought_reward": cfg.get("tp_neutral_thought_reward"),
        "tp_correct_path_bonus": cfg.get("tp_correct_path_bonus"),
        "tp_wrong_final_penalty": cfg.get("tp_wrong_final_penalty"),
        "tp_outcome_bonus_scale": cfg.get("tp_outcome_bonus_scale"),
        "tp_only_penalize_error_thoughts": cfg.get("tp_only_penalize_error_thoughts"),
        "tp_leave_unmatched_incorrect_zero": cfg.get("tp_leave_unmatched_incorrect_zero"),
        "tp_group_std_eps": cfg.get("tp_group_std_eps"),
        "prm_tp_pos_threshold": cfg.get("prm_tp_pos_threshold"),
        "prm_tp_neg_threshold": cfg.get("prm_tp_neg_threshold"),

        # Validation config
        "run_validation": cfg.get("run_validation"),
        "validation_size": cfg.get("validation_size"),
        "validation_seed": cfg.get("validation_seed"),
        "validation_max_examples": cfg.get("validation_max_examples"),
        "validation_temperature": cfg.get("validation_temperature"),
        "validation_top_p": cfg.get("validation_top_p"),
    })

    row.update({
        # Parameter counts
        "total_params": params.get("total_params"),
        "trainable_params_lora": params.get("trainable_params_lora"),
        "backbone_params": params.get("backbone_params"),
        "reward_model_params": params.get("reward_model_params"),
        "prm_params": params.get("prm_params"),
    })

    row.update({
        # FLOP accounting config
        "N_params": comp.get("N_params"),
        "R_params": comp.get("R_params"),
        "c_forward": comp.get("c_forward"),
        "c_backward": comp.get("c_backward"),
        "f_lora": comp.get("f_lora"),
        "update_backbone_fraction_a": comp.get("update_backbone_fraction_a"),
        "update_effective_param_fraction": comp.get("update_effective_param_fraction"),

        # FLOP totals
        "rollout_tokens": totals.get("rollout_tokens"),
        "update_tokens": totals.get("update_tokens"),
        "rollout_flops": totals.get("rollout_flops"),
        "update_flops": totals.get("update_flops"),
        "total_flops": totals.get("total_flops"),
    })

    total_flops = to_float(row.get("total_flops"))
    rollout_flops = to_float(row.get("rollout_flops"))
    update_flops = to_float(row.get("update_flops"))

    if total_flops and total_flops > 0:
        row["rho_update_total"] = update_flops / total_flops if update_flops is not None else None
        row["rho_rollout_total"] = rollout_flops / total_flops if rollout_flops is not None else None
        row["log10_total_flops"] = math.log10(total_flops)
    else:
        row["rho_update_total"] = None
        row["rho_rollout_total"] = None
        row["log10_total_flops"] = None

    search_learning = None
    if rollout_flops is not None and update_flops is not None:
        search_learning = rollout_flops + update_flops

    if search_learning and search_learning > 0:
        row["rho_update_search_learning"] = update_flops / search_learning
        row["rho_rollout_search_learning"] = rollout_flops / search_learning
    else:
        row["rho_update_search_learning"] = None
        row["rho_rollout_search_learning"] = None

    row.update({
        # Validation metrics
        "validation_num_examples": validation.get("num_examples"),
        "validation_accuracy_exact": validation.get("accuracy_exact"),
        "validation_format_ok_rate": validation.get("format_ok_rate"),
        "validation_has_reasoning_rate": validation.get("has_reasoning_rate"),
        "validation_mean_completion_tokens": validation.get("mean_completion_tokens"),
        "validation_hit_max_length_rate": validation.get("hit_max_length_rate"),
    })

    # Flatten all reward logger statistics
    for reward_name, stats in rewards.items():
        if isinstance(stats, dict):
            for stat_name, value in stats.items():
                row[f"reward_{reward_name}_{stat_name}"] = value

    reward_mode = row.get("reward_mode")

    row["primary_reward_mean"] = reward_mean(rewards, "total_reward")
    row["primary_reward_std"] = reward_std(rewards, "total_reward")
    row["primary_reward_min"] = safe_get(rewards, "total_reward", "min")
    row["primary_reward_max"] = safe_get(rewards, "total_reward", "max")

    row["correctness_train_proxy_mean"] = choose_correctness_mean(reward_mode, rewards)

    # Prefer validation accuracy for actual final-answer correctness when available.
    row["accuracy_main"] = (
        row.get("validation_accuracy_exact")
        if row.get("validation_accuracy_exact") is not None
        else row.get("correctness_train_proxy_mean")
    )

    # Convenience reward-specific columns
    row["sparse_correct_mean"] = reward_mean(rewards, "sparse_correct")
    row["structured_check_answer_mean"] = reward_mean(rewards, "check_answer")
    row["structured_check_numbers_mean"] = reward_mean(rewards, "check_numbers")
    row["dense_correct_mean"] = reward_mean(rewards, "dense_correct")
    row["dense_closeness_mean"] = reward_mean(rewards, "dense_closeness")
    row["verifier_correct_mean"] = reward_mean(rewards, "verifier_correct")
    row["verifier_proc_mean"] = reward_mean(rewards, "verifier_proc")
    row["prm_correct_mean"] = reward_mean(rewards, "prm_correct")
    row["prm_proc_mean"] = reward_mean(rewards, "prm_proc")
    row["prm_num_scored_steps_mean"] = reward_mean(rewards, "prm_num_scored_steps")
    row["prm_tp_correct_mean"] = reward_mean(rewards, "prm_tp_correct")
    row["prm_tp_proc_mean"] = reward_mean(rewards, "prm_tp_proc")
    row["prm_tp_num_thoughts_mean"] = reward_mean(rewards, "prm_tp_num_thoughts")
    row["prm_tp_num_correct_thoughts_mean"] = reward_mean(rewards, "prm_tp_num_correct_thoughts")
    row["prm_tp_num_incorrect_thoughts_mean"] = reward_mean(rewards, "prm_tp_num_incorrect_thoughts")

    return row


def find_stage_summaries(stage_root: Path) -> List[Path]:
    if not stage_root.exists():
        return []
    return sorted(stage_root.rglob("summary.json"))


def load_rows(stage: str, stage_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for summary_path in find_stage_summaries(stage_root):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            rows.append(flatten_summary(summary, summary_path, stage))
        except Exception as e:
            rows.append({
                "stage": stage,
                "summary_path": str(summary_path),
                "load_error": repr(e),
            })

    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else []

    with open(path, "w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect Stage A/B/C/D summary.json files into flat CSV/JSONL files."
    )
    ap.add_argument("--base_dir", type=str, default=".")
    ap.add_argument("--out_dir", type=str, default="collected_stage_summaries")
    ap.add_argument(
        "--stages",
        nargs="*",
        default=["stageA", "stageB", "stageC", "stageD"],
        choices=list(STAGE_ROOTS.keys()),
    )
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {
        "base_dir": str(base_dir),
        "stages": {},
        "outputs": {},
    }

    for stage in args.stages:
        stage_root = base_dir / STAGE_ROOTS[stage]
        rows = load_rows(stage, stage_root)
        all_rows.extend(rows)

        jsonl_path = out_dir / f"{stage}_collected.jsonl"
        csv_path = out_dir / f"{stage}_collected.csv"

        write_jsonl(jsonl_path, rows)
        write_csv(csv_path, rows)

        manifest["stages"][stage] = {
            "root": str(stage_root),
            "num_rows": len(rows),
        }
        manifest["outputs"][stage] = {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
        }

    combined_jsonl = out_dir / "all_stages_collected.jsonl"
    combined_csv = out_dir / "all_stages_collected.csv"

    write_jsonl(combined_jsonl, all_rows)
    write_csv(combined_csv, all_rows)

    manifest["outputs"]["all"] = {
        "jsonl": str(combined_jsonl),
        "csv": str(combined_csv),
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps({
        "out_dir": str(out_dir),
        "stage_counts": {
            stage: manifest["stages"][stage]["num_rows"]
            for stage in manifest["stages"]
        },
        "combined_rows": len(all_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
