#!/usr/bin/env python3
# rl_posttrain_train_tpprm.py
#
# One-file GRPO post-training script for compute-allocation / reward-scaling experiments.
#
# This keeps the overall "one-script" logic of your original file, but upgrades the PRM path
# to a thought-level, capability-adaptive TP-PRM-style reward that is much closer in spirit
# to paper b) than simple step-score averaging.
#
# Important honesty note:
# - Because this still uses TRL's standard GRPOTrainer entrypoint, the PRM path below remains
#   a strong "thought-level PRM reward for GRPO" implementation rather than a line-for-line
#   reproduction of the authors' custom TP-GRPO trainer.
# - It is, however, a much stronger compute-allocation condition than your previous scalar
#   average-of-step-scores PRM baseline and is suitable as the main PRM condition in a
#   one-file ablation study.
#
# Key PRM upgrades relative to your old script:
# - thought-level segmentation instead of raw step averaging
# - contiguous merge of same-sign thoughts
# - verifier-driven thought scoring
# - correct-path vs incorrect-path differentiated reward shaping
# - capability-adaptive weighting using group accuracy
# - stronger logging of thought annotations
#
# Recommended naming in paper:
# - reward_mode=prm_scalar   -> old baseline-style PRM
# - reward_mode=prm_tp       -> main thought-level PRM condition
#
# Example:
# python rl_posttrain_train_tpprm.py \
#   --model Qwen/Qwen2.5-1.5B-Instruct \
#   --out runs_tpprm_pilot \
#   --steps 100 \
#   --K 4 \
#   --L 512 \
#   --grad_accum 2 \
#   --reward_mode prm_tp \
#   --verifier_model_name Qwen/Qwen2.5-Math-PRM-7B \
#   --kl_beta 0.001 \
#   --temperature 0.7 \
#   --validation_size 64 \
#   --validation_max_examples 64

import os
import re
import json
import random
import argparse
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModel, AutoTokenizer

import unsloth
from unsloth import FastLanguageModel

try:
    import sympy as sp  # type: ignore
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    HAS_SYMPY = True
except Exception:
    sp = None
    parse_expr = None
    standard_transformations = ()
    implicit_multiplication_application = None
    convert_xor = None
    HAS_SYMPY = False


# --------------------------- Config ---------------------------

class Cfg:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_seq_length: int = 4096

    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = (
        "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj".split()
    )
    full_finetuning: bool = False

    output_dir: str = "runs/g1b_A"
    max_steps: int = 100
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    num_generations: int = 2
    max_prompt_length: int = 512
    max_completion_length: int = 256
    temperature: float = 0.5
    logging_steps: int = 10
    save_steps: int = 500

    reward_mode: str = "structured"

    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0
    reward_standardize: bool = False

    reward_mode_scale: Dict[str, float] = {
        "sparse": 1.0,
        "structured": 1.0 / 6.0,
        "dense": 1.0,
        "dense_verifier": 1.0,
        "prm_scalar": 2.0,
        "prm_tp": 1.0,
    }
    reward_mode_shift: Dict[str, float] = {
        "sparse": 0.0,
        "structured": 0.0,
        "dense": 0.0,
        "dense_verifier": 0.0,
        "prm_scalar": -0.5,
        "prm_tp": 0.0,
    }

    dense_err_scale: float = 5.0

    # legacy scalar-PRM baseline
    prm_model_name: str = ""
    prm_device: str = "cuda"
    prm_alpha: float = 0.8
    prm_max_steps_scored: int = 64
    prm_include_outcome: bool = True
    prm_outcome_scale: float = 1.0
    prm_system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."

    # new TP-PRM style verifier
    verifier_model_name: str = ""
    verifier_device: str = "cuda"
    verifier_max_new_tokens: int = 8
    prm_tp_pos_threshold: float = 0.75
    prm_tp_neg_threshold: float = 0.25

    # thought-level process reward parameters
    thought_sep_mode: str = "blankline"
    tp_merge_same_sign: bool = True
    tp_capability_adaptive: bool = True
    tp_correct_thought_reward: float = 1.0
    tp_incorrect_thought_penalty: float = -1.0
    tp_neutral_thought_reward: float = 0.0
    tp_correct_path_bonus: float = 0.15
    tp_wrong_final_penalty: float = -0.15
    tp_outcome_bonus_scale: float = 0.25
    tp_only_penalize_error_thoughts: bool = True
    tp_leave_unmatched_incorrect_zero: bool = True
    tp_group_std_eps: float = 1e-6

    kl_beta: float = 0.001

    log_kl_proxy: bool = False
    ref_model_name: str = ""

    verifier_strict: bool = False

    flops_scale_forward: float = 2.0
    flops_scale_backward: float = 6.0
    reward_model_params: int = 0
    update_backbone_fraction: float = 0.85

    reuse_prefill_across_K: bool = False

    dataset_name: str = "polaris53k"
    dataset_config: Optional[str] = None
    train_split: str = "train"

    reasoning_start: str = "<start_working_out>"
    reasoning_end: str = "<end_working_out>"
    solution_start: str = "<SOLUTION>"
    solution_end: str = "</SOLUTION>"

    sparse_answer_prefix: str = "Answer:"

    verifier_friendly_prompt: bool = False

    logs_dir: str = "logs"
    traces_filename: str = "rl_traces.jsonl"
    summary_filename: str = "summary.json"

    seed: int = 42
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    gpu_memory_utilization: float = 0.7

    validation_size: int = 1000
    validation_seed: int = 12345
    run_validation: bool = True
    validation_max_examples: int = 1000
    validation_temperature: float = 0.0
    validation_top_p: float = 1.0
    validation_predictions_filename: str = "validation_predictions.jsonl"
    split_manifest_filename: str = "split_manifest.json"

    debug_completion_preview_chars: int = 2000
    debug_print_dataset_examples: int = 3


CFG = Cfg()

if "LORA_R" in os.environ:
    CFG.lora_r = int(os.environ["LORA_R"])
    CFG.lora_alpha = 2 * CFG.lora_r
    print(f"[CFG] Overriding LoRA rank to r={CFG.lora_r}, alpha={CFG.lora_alpha}")


# ------------------------ Utilities --------------------------

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def mean_logp(model, tokenizer, texts: List[str], device: str) -> List[float]:
    model.eval()
    out: List[float] = []
    for t in texts:
        toks = tokenizer(
            t,
            return_tensors="pt",
            truncation=True,
            max_length=CFG.max_seq_length,
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        input_ids = toks["input_ids"]
        attn = toks.get("attention_mask", None)

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        logp = F.log_softmax(shift_logits, dim=-1)
        tok_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        if attn is not None:
            m = attn[:, 1:].float()
            denom = float(m.sum().item()) if float(m.sum().item()) > 0 else float(tok_logp.numel())
            avg = float((tok_logp * m).sum().item() / denom)
        else:
            avg = float(tok_logp.mean().item())
        out.append(avg)
    return out


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(os.path.join(CFG.output_dir, CFG.logs_dir), exist_ok=True)


def path_in_logs(fname: str) -> str:
    return os.path.join(CFG.output_dir, CFG.logs_dir, fname)


class RewardLogger:
    def __init__(self, reward_names: List[str]):
        self.reward_names = reward_names
        self.reset()

    def reset(self):
        self.count = 0
        self.sums = defaultdict(float)
        self.sumsq = defaultdict(float)
        self.mins = defaultdict(lambda: float("inf"))
        self.maxs = defaultdict(lambda: float("-inf"))

    def add_batch(self, per_reward: Dict[str, List[float]]):
        n = None
        for k, vals in per_reward.items():
            if not isinstance(vals, list):
                continue
            if n is None:
                n = len(vals)
            for v in vals:
                fv = float(v)
                self.sums[k] += fv
                self.sumsq[k] += fv * fv
                self.mins[k] = min(self.mins[k], fv)
                self.maxs[k] = max(self.maxs[k], fv)
        if n is not None:
            self.count += int(n)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in self.reward_names:
            n = max(self.count, 1)
            mean = self.sums[k] / n
            var = max(self.sumsq[k] / n - mean * mean, 0.0)
            out[k] = {
                "count": int(self.count),
                "mean": float(mean),
                "std": float(var ** 0.5),
                "min": float(self.mins[k] if self.mins[k] != float("inf") else 0.0),
                "max": float(self.maxs[k] if self.maxs[k] != float("-inf") else 0.0),
            }
        return out


# --------------------------- Answer normalization ----------------------------

CURRENT_A_KEY = "answer"


def _extract_balanced_braces(s: str, start_idx: int) -> Tuple[Optional[str], int]:
    if start_idx >= len(s) or s[start_idx] != "{":
        return None, start_idx
    depth = 0
    out = []
    i = start_idx
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
            if depth > 1:
                out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out), i + 1
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return None, start_idx


def _strip_outer_braces(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    ok = False
                    break
        if ok:
            s = s[1:-1].strip()
        else:
            break
    return s


def _replace_latex_sqrt(s: str) -> str:
    while "\\sqrt" in s:
        idx = s.find("\\sqrt")
        j = idx + len("\\sqrt")
        while j < len(s) and s[j].isspace():
            j += 1
        if j < len(s) and s[j] == "{":
            inside, j2 = _extract_balanced_braces(s, j)
            if inside is None:
                break
            repl = f"sqrt({_replace_latex_sqrt(inside)})"
            s = s[:idx] + repl + s[j2:]
        else:
            break
    return s


def _cleanup_math_text(s: str) -> str:
    s = s.strip()
    s = s.replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("×", "*").replace("·", "*")
    s = s.replace("^", "**")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\pi", "pi").replace("\\%", "%")
    s = _replace_latex_sqrt(s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[.,;:]\s*$", "", s).strip()
    return _strip_outer_braces(s)


def _extract_boxed_content(text: str) -> Optional[str]:
    if text is None:
        return None
    m = re.search(r"\\boxed\s*\{", text)
    if not m:
        return None
    content, _ = _extract_balanced_braces(text, m.end() - 1)
    if content is None:
        return None
    return content.strip()


def _extract_answer_line(text: str) -> Optional[str]:
    if text is None:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        low = ln.lower()
        if low.startswith("answer:") or low.startswith("final answer:"):
            return ln.split(":", 1)[1].strip()
    return None


def _safe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
        try:
            return float(s)
        except Exception:
            return None
    if re.fullmatch(r"[-+]?\d+/\d+", s):
        try:
            a, b = s.split("/")
            return float(a) / float(b)
        except Exception:
            return None
    return None


def _sympy_expr_from_text(s: str):
    if not HAS_SYMPY or parse_expr is None:
        return None
    raw = _cleanup_math_text(s)
    if not raw:
        return None

    placeholders = {
        "sqrt": "__SQRT__",
        "sin": "__SIN__",
        "cos": "__COS__",
        "tan": "__TAN__",
        "log": "__LOG__",
        "ln": "__LN__",
        "pi": "__PI__",
    }
    temp = raw
    for k, v in placeholders.items():
        temp = temp.replace(k, v)

    temp = re.sub(r"(\d)([A-Za-z(])", r"\1*\2", temp)
    temp = re.sub(r"(\))([A-Za-z(])", r"\1*\2", temp)
    temp = re.sub(r"([A-Za-z])(\()", r"\1*\2", temp)

    for k, v in placeholders.items():
        temp = temp.replace(v, k)

    temp = temp.replace("{", "(").replace("}", ")")
    temp = temp.replace("[", "(").replace("]", ")")

    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    local_dict = {ch: sp.Symbol(ch) for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    local_dict.update({
        "pi": sp.pi,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "log": sp.log,
        "ln": sp.log,
    })
    try:
        expr = parse_expr(temp, transformations=transformations, local_dict=local_dict, evaluate=True)
        expr = sp.simplify(expr)
        return expr
    except Exception:
        return None


def _looks_symbolic_math(s: str) -> bool:
    if not s:
        return False
    s = str(s)
    if any(tok in s for tok in [
        "\\frac", "\\dfrac", "\\tfrac", "\\sqrt",
        "/", "sqrt(", "pi", "+", "-", "*", "**", "(", ")", "="
    ]):
        return True
    if re.search(r"[A-Za-z]{2,}", s):
        return True
    return False


def canonical_answer(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    boxed = _extract_boxed_content(s)
    if boxed is not None:
        s = boxed
    s = _cleanup_math_text(s)
    if not s:
        return None

    f = _safe_float(s)
    if f is not None:
        if abs(f - round(f)) < 1e-12:
            return str(int(round(f)))
        return str(float(f))

    expr = _sympy_expr_from_text(s) if _looks_symbolic_math(s) else None
    if expr is not None:
        try:
            return str(sp.simplify(expr))
        except Exception:
            pass
    return s


def answers_equal(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False

    cp = canonical_answer(pred)
    cg = canonical_answer(gold)
    if cp is None or cg is None:
        return False
    if cp == cg:
        return True

    if HAS_SYMPY:
        ep = _sympy_expr_from_text(pred)
        eg = _sympy_expr_from_text(gold)
        if ep is not None and eg is not None:
            try:
                return bool(sp.simplify(ep - eg) == 0)
            except Exception:
                pass

    fp = _safe_float(cp)
    fg = _safe_float(cg)
    if fp is not None and fg is not None:
        return abs(fp - fg) <= 1e-9
    return False


def answer_for_numeric_shaping(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    cx = canonical_answer(x)
    if cx is None:
        return None
    return _safe_float(cx)


# --------------------------- Data ----------------------------

DATASETS = {
    "polaris53k": {
        "hf_name": "POLARIS-Project/Polaris-Dataset-53K",
        "config": None,
        "train_split": "train",
        "test_split": None,
        "q_key": "problem",
        "a_key": "answer",
        "answer_parser": "polaris_final",
    },
    "math500": {
        "hf_name": "HuggingFaceH4/MATH-500",
        "config": None,
        "train_split": "train",
        "test_split": "test",
        "q_key": "problem",
        "a_key": "solution",
        "answer_parser": "math_boxed_or_raw",
    },
    "aime2024": {
        "hf_name": "your_org/AIME2024",
        "config": None,
        "train_split": "train",
        "test_split": "test",
        "q_key": "problem",
        "a_key": "answer",
        "answer_parser": "aime_int",
    },
    "gsm8k": {
        "hf_name": "openai/gsm8k",
        "config": "main",
        "train_split": "train",
        "test_split": "test",
        "q_key": "question",
        "a_key": "answer",
        "answer_parser": "gsm8k_hash",
    },
}


def extract_gsm8k_hash_answer(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####")[-1].strip()


def parse_answer(example: dict, parser: str) -> Optional[str]:
    raw = example.get(CURRENT_A_KEY, None)
    if raw is None:
        return None
    raw = str(raw)

    if parser == "gsm8k_hash":
        return extract_gsm8k_hash_answer(raw)
    if parser == "math_boxed_or_raw":
        boxed = _extract_boxed_content(raw)
        return boxed.strip() if boxed is not None else raw.strip()
    if parser == "aime_int":
        m = re.findall(r"-?\d+", raw)
        return m[-1].strip() if m else None
    if parser == "polaris_final":
        boxed = _extract_boxed_content(raw)
        return boxed.strip() if boxed is not None else raw.strip()

    return raw.strip()


def make_train_val_datasets(tokenizer):
    ds_key = CFG.dataset_name.strip().lower()
    spec = DATASETS.get(ds_key, None)
    if spec is None:
        raise ValueError(f"Unknown dataset_name={CFG.dataset_name}. Use one of: {list(DATASETS.keys())}")

    hf_name = spec["hf_name"]
    cfg = spec["config"]
    split = spec.get("train_split", CFG.train_split)

    global CURRENT_A_KEY
    CURRENT_A_KEY = spec["a_key"]

    print(f"[Data] Loading {ds_key}: {hf_name} split={split}")
    ds = load_dataset(hf_name, cfg, split=split) if cfg else load_dataset(hf_name, split=split)

    q_key = spec["q_key"]
    parser = spec["answer_parser"]

    rs, re_, ss, se = CFG.reasoning_start, CFG.reasoning_end, CFG.solution_start, CFG.solution_end
    extra = ""
    if CFG.verifier_friendly_prompt:
        extra = (
            "In the working out, write key calculations as separate lines of the form "
            "'expression = value' so they can be automatically verified.\n"
        )

    if CFG.reward_mode.lower().strip() == "sparse":
        system_prompt = (
            "Solve the math problem. "
            f"End your response with exactly one final line in this format:\n"
            f"{CFG.sparse_answer_prefix} \\boxed{{your_answer}}\n"
        )
    else:
        system_prompt = (
            "You are given a math problem. Think step by step and provide your working out. "
            + extra +
            f"Place the working between {rs} and {re_}. "
            "Separate major reasoning steps with a blank line. "
            f"Then provide your final answer between {ss} and {se}. "
            "Also include the final answer in \\boxed{} form."
        )

    def map_ex(example, idx):
        question = str(example[q_key]).strip()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        ans = parse_answer(example, parser)
        row = {
            "example_id": int(idx),
            "prompt": prompt_text,
            "question": question,
            "answer": ans,
        }
        if "difficulty" in example and example["difficulty"] is not None:
            row["difficulty"] = str(example["difficulty"])
        return row

    mapped = ds.map(map_ex, with_indices=True, remove_columns=ds.column_names)
    mapped = mapped.filter(lambda r: r["answer"] is not None and str(r["answer"]).strip() != "")
    mapped = mapped.shuffle(seed=CFG.validation_seed)

    n_total = len(mapped)
    if n_total < 2:
        raise RuntimeError(f"Dataset too small after filtering: {n_total}")

    n_val = int(max(0, min(CFG.validation_size, n_total - 1)))
    if n_val == 0:
        train = mapped.with_format("python")
        val = mapped.select([]).with_format("python")
    else:
        val = mapped.select(range(n_val)).with_format("python")
        train = mapped.select(range(n_val, n_total)).with_format("python")

    print(f"[Data] train={len(train)} | val={len(val)} | total_filtered={n_total}")

    dbg_n = min(CFG.debug_print_dataset_examples, len(train))
    for i in range(dbg_n):
        ex = train[i]
        print(f"\n[DATA DEBUG {i}]")
        print("QUESTION:", ex["question"][:300])
        print("ANSWER_RAW:", ex["answer"])
        print("ANSWER_CANON:", canonical_answer(ex["answer"]))

    return train, val


# -------------------- FLOPs Accountant -----------------------

class FlopsAccountant:
    def __init__(self, N: int, R: int, c_fwd: float, c_bwd: float, bwd_frac: float, a_backbone: float = 0.85):
        self.N, self.R = int(N), int(R)
        self.c_fwd, self.c_bwd = float(c_fwd), float(c_bwd)
        self.f_lora = float(bwd_frac)
        self.a_backbone = max(0.0, min(1.0, float(a_backbone)))
        self.eff_frac = self.a_backbone + (1.0 - self.a_backbone) * self.f_lora
        self.totals = defaultdict(float)

    def add(self, rollout_tokens: int, update_tokens: int):
        rollout_flops = (self.c_fwd * (self.N + self.R)) * float(rollout_tokens)
        update_flops = (self.c_bwd * (self.N * self.eff_frac)) * float(update_tokens)
        self.totals["rollout_tokens"] += float(rollout_tokens)
        self.totals["update_tokens"] += float(update_tokens)
        self.totals["rollout_flops"] += float(rollout_flops)
        self.totals["update_flops"] += float(update_flops)
        self.totals["total_flops"] += float(rollout_flops + update_flops)

    def summary(self) -> Dict[str, Any]:
        return {
            "N_params": self.N,
            "R_params": self.R,
            "c_forward": self.c_fwd,
            "c_backward": self.c_bwd,
            "f_lora": self.f_lora,
            "update_backbone_fraction_a": self.a_backbone,
            "update_effective_param_fraction": self.eff_frac,
            "totals": dict(self.totals),
        }


# ----------------------- Verifier / TP-PRM -----------------------


class RealPRMVerifier:
    """
    Real process-reward-model verifier compatible with Qwen2.5-Math-PRM models.

    Core semantics follow Qwen's PRM usage:
    - reasoning steps / thoughts are separated with "<extra_0>"
    - we run AutoModel (not AutoModelForCausalLM)
    - we extract the probability of the positive label at each separator token

    This class supports both:
    - scalar PRM baseline (average step score)
    - TP-style thought-level PRM (label each thought by its PRM score)
    """

    def __init__(self, model_name: str, device: str):
        self.model_name = (model_name or "").strip()
        self.device = device
        self.model = None
        self.tokenizer = None
        self.num_params = 0
        self.step_sep_token = "<extra_0>"
        self.step_sep_id = None

        if self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True,
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if device == "cuda" else None,
            ).eval()
            self.model.use_cache=False
            if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
                self.model.generation_config.use_cache = False
            if device == "cpu":
                self.model.to("cpu")
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.num_params = int(sum(p.numel() for p in self.model.parameters()))
            sep_ids = self.tokenizer.encode(self.step_sep_token, add_special_tokens=False)
            if len(sep_ids) != 1:
                raise RuntimeError(
                    f"Expected {self.step_sep_token} to map to a single token for PRM scoring, got {sep_ids}"
                )
            self.step_sep_id = int(sep_ids[0])

    def _model_device(self):
        if self.model is None:
            return torch.device("cpu")
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @torch.no_grad()
    def step_scores(self, question: str, steps: List[str], system_prompt: Optional[str] = None) -> List[float]:
        if self.model is None or self.tokenizer is None or self.step_sep_id is None:
            return []
        cleaned_steps = [s.strip() for s in steps if s and s.strip()]
        if not cleaned_steps:
            return []

        system_prompt = system_prompt or CFG.prm_system_prompt
        assistant_content = self.step_sep_token.join(cleaned_steps) + self.step_sep_token
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
        conversation_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        input_ids = self.tokenizer.encode(
            conversation_str,
            return_tensors="pt",
            truncation=True,
            max_length=CFG.max_seq_length,
        ).to(self._model_device())

        outputs = self.model(input_ids=input_ids, use_cache= False)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        if logits.shape[-1] != 2:
            raise RuntimeError(
                f"Expected PRM logits with 2 labels, got shape {tuple(logits.shape)}. "
                "Use a real PRM such as Qwen/Qwen2.5-Math-PRM-7B or 72B."
            )

        probs = torch.softmax(logits, dim=-1)
        token_mask = (input_ids == self.step_sep_id)
        sep_probs = probs[0][token_mask[0]]
        if sep_probs.numel() == 0:
            return []
        return [float(x) for x in sep_probs[:, 1].detach().float().cpu().tolist()]

    def score_thought(
        self,
        question: str,
        gold_answer: Optional[str],
        previous_thoughts: List[str],
        current_thought: str,
        final_prediction: Optional[str],
        final_is_correct: bool,
    ) -> Tuple[float, str]:
        """
        Score the current thought using real PRM semantics:
        run the PRM on previous_thoughts + current_thought, then take the last step score.
        """
        # lightweight symbolic shortcut remains useful for exact equation lines
        eq_pat = re.compile(r"^\s*(.+?)\s*=\s*(.+?)\s*$")
        lines = [ln.strip() for ln in current_thought.splitlines() if ln.strip()]
        eq_scores = []
        if HAS_SYMPY:
            for ln in lines:
                mm = eq_pat.match(ln)
                if mm is None:
                    continue
                lhs, rhs = mm.group(1), mm.group(2)
                lhs_e = _sympy_expr_from_text(lhs)
                rhs_e = _sympy_expr_from_text(rhs)
                if lhs_e is None or rhs_e is None:
                    continue
                try:
                    eq_scores.append(1.0 if sp.simplify(lhs_e - rhs_e) == 0 else -1.0)
                except Exception:
                    pass
        if eq_scores:
            mean_eq = sum(eq_scores) / len(eq_scores)
            if mean_eq > 0.5:
                return 1.0, "correct"
            if mean_eq < -0.5:
                return -1.0, "incorrect"

        scores = self.step_scores(
            question=question,
            steps=list(previous_thoughts) + [current_thought],
            system_prompt=CFG.prm_system_prompt,
        )
        if not scores:
            if final_is_correct:
                return 0.5, "correct"
            return 0.0, "unsure"

        s = float(scores[-1])
        if s >= CFG.prm_tp_pos_threshold:
            return s, "correct"
        if s <= CFG.prm_tp_neg_threshold:
            return s - 1.0, "incorrect"
        return s - 0.5, "unsure"


def merge_same_sign_thoughts(thoughts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not thoughts:
        return []
    merged = [dict(thoughts[0])]
    for th in thoughts[1:]:
        prev = merged[-1]
        if th["label"] == prev["label"]:
            prev["text"] = (prev["text"].rstrip() + "\n\n" + th["text"].lstrip()).strip()
            prev["score_raw"] = float((prev["score_raw"] + th["score_raw"]) / 2.0)
            prev["num_atomic"] = int(prev.get("num_atomic", 1) + th.get("num_atomic", 1))
        else:
            merged.append(dict(th))
    return merged


def normalize_group(xs: List[float], eps: float) -> List[float]:
    if not xs:
        return []
    m = sum(xs) / float(len(xs))
    v = sum((x - m) ** 2 for x in xs) / float(len(xs))
    s = math.sqrt(max(v, 0.0))
    if s < eps:
        return [0.0 for _ in xs]
    return [(x - m) / s for x in xs]


def suffix_weighted_sum(scores: List[float]) -> float:
    # proxy for TP-GRPO cumulative tail-advantage in scalar-reward GRPO setting
    if not scores:
        return 0.0
    out = 0.0
    running = 0.0
    for x in reversed(scores):
        running += x
        out += running
    return out / float(len(scores))


# ---------------- Rewards + Accounting ----------------

def build_rewards_and_accounting(
    tokenizer,
    accountant: FlopsAccountant,
    reward_logger: RewardLogger,
    train_model=None,
    ref_model=None,
    prm_model=None,
    prm_tokenizer=None,
    verifier=None,
):
    rs, re_, ss, se = CFG.reasoning_start, CFG.reasoning_end, CFG.solution_start, CFG.solution_end

    match_format = re.compile(
        rf"^[\s]{{0,}}{re.escape(rs)}.+?{re.escape(re_)}.*?{re.escape(ss)}(.+?){re.escape(se)}[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL,
    )
    eq_pat = re.compile(r"^\s*(.+?)\s*=\s*(.+?)\s*$")

    def extract_solution_strict(text: str) -> Optional[str]:
        m = match_format.search(text)
        return None if m is None else m.group(1).strip()

    def extract_sparse_answer(text: str) -> Optional[str]:
        if text is None:
            return None
        ans_line = _extract_answer_line(text)
        if ans_line:
            return ans_line.strip()
        boxed = _extract_boxed_content(text)
        if boxed is not None:
            return boxed.strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            last = lines[-1]
            if len(last) <= 80:
                return last
        return None

    def extract_solution_str_loose(text: str) -> Optional[str]:
        if text is None:
            return None
        if CFG.reward_mode.lower().strip() == "sparse":
            return extract_sparse_answer(text)
        m = re.search(rf"{re.escape(ss)}(.*?){re.escape(se)}", text, flags=re.DOTALL)
        if m is not None:
            return m.group(1).strip()
        boxed = _extract_boxed_content(text)
        if boxed is not None:
            return boxed.strip()
        ans_line = _extract_answer_line(text)
        if ans_line:
            return ans_line.strip()
        return None

    def extract_solution_num_loose(text: str) -> Optional[float]:
        sol = extract_solution_str_loose(text)
        return answer_for_numeric_shaping(sol)

    def extract_reasoning_block(text: str) -> Optional[str]:
        m = re.search(rf"{re.escape(rs)}(.*?){re.escape(re_)}", text, flags=re.DOTALL)
        return None if m is None else m.group(1)

    def sparse_correctness(completions, answer, **kwargs) -> Dict[str, List[float]]:
        corr, tot = [], []
        for resp, true_answer in zip(completions, answer):
            sol = extract_sparse_answer(resp)
            ok = answers_equal(sol, true_answer)
            corr.append(1.0 if ok else 0.0)
            tot.append(1.0 if ok else 0.0)
        return {"sparse_correct": corr, "total_reward": tot}

    def structured_codeA(completions, answer, **kwargs) -> Dict[str, List[float]]:
        fmt_exact, fmt_approx, ans_score, num_score = [], [], [], []
        for resp, true_answer in zip(completions, answer):
            fmt_exact.append(3.0 if match_format.search(resp) is not None else 0.0)
            sc = 0.0
            sc += 0.5 if resp.count(rs) == 1 else -0.5
            sc += 0.5 if resp.count(re_) == 1 else -0.5
            sc += 0.5 if resp.count(ss) == 1 else -0.5
            sc += 0.5 if resp.count(se) == 1 else -0.5
            fmt_approx.append(sc)

            sol = extract_solution_str_loose(resp)
            if sol is None:
                ans_score.append(0.0)
            else:
                if answers_equal(sol, true_answer):
                    ans_score.append(3.0)
                else:
                    g = answer_for_numeric_shaping(sol)
                    t = answer_for_numeric_shaping(true_answer)
                    if g is not None and t is not None and abs(t) > 1e-12:
                        ratio = g / t
                        if 0.9 <= ratio <= 1.1:
                            ans_score.append(0.5)
                        elif 0.8 <= ratio <= 1.2:
                            ans_score.append(0.25)
                        else:
                            ans_score.append(-1.0)
                    else:
                        ans_score.append(-0.5)

            gnum = answer_for_numeric_shaping(sol)
            tnum = answer_for_numeric_shaping(true_answer)
            if gnum is None or tnum is None:
                num_score.append(0.0)
            else:
                num_score.append(1.5 if abs(gnum - tnum) <= 1e-9 else 0.0)

        total = [float(a + b + c + d) for a, b, c, d in zip(fmt_exact, fmt_approx, ans_score, num_score)]
        return {
            "match_format_exactly": fmt_exact,
            "match_format_approximately": fmt_approx,
            "check_answer": ans_score,
            "check_numbers": num_score,
            "total_reward": total,
        }

    def dense_bounded(completions, answer, **kwargs) -> Dict[str, List[float]]:
        fmt, closeness, correct, total = [], [], [], []
        scale = float(CFG.dense_err_scale)
        for resp, true_answer in zip(completions, answer):
            ta = answer_for_numeric_shaping(true_answer)
            gnum = extract_solution_num_loose(resp)

            f = 0.0
            f += 0.05 if resp.count(rs) == 1 else -0.05
            f += 0.05 if resp.count(re_) == 1 else -0.05
            f += 0.05 if resp.count(ss) == 1 else -0.05
            f += 0.05 if resp.count(se) == 1 else -0.05
            fmt.append(float(f))

            if ta is None or gnum is None:
                c = -1.0
            else:
                err = abs(gnum - ta)
                c01 = math.exp(-err / max(scale, 1e-6))
                c = 2.0 * c01 - 1.0
            closeness.append(float(c))

            sol = extract_solution_str_loose(resp)
            ok = answers_equal(sol, true_answer)
            correct.append(1.0 if ok else 0.0)
            spike = 0.3 if ok else 0.0

            tr = c + f + spike
            tr = max(-1.0, min(1.0, tr))
            total.append(float(tr))

        return {
            "dense_format": fmt,
            "dense_closeness": closeness,
            "dense_correct": correct,
            "total_reward": total,
        }

    def verifier_process_score(resp: str) -> float:
        if not HAS_SYMPY:
            raise RuntimeError("dense_verifier mode requires sympy. Install: pip install sympy")
        rb = extract_reasoning_block(resp)
        if rb is None:
            return -1.0
        lines = [ln.strip() for ln in rb.splitlines() if ln.strip()]
        if not lines:
            return -1.0

        scores: List[float] = []
        for ln in lines:
            mm = eq_pat.match(ln)
            if mm is None:
                if CFG.verifier_strict:
                    scores.append(-1.0)
                continue
            lhs, rhs = mm.group(1), mm.group(2)
            try:
                lhs_e = _sympy_expr_from_text(lhs)
                rhs_e = _sympy_expr_from_text(rhs)
                if lhs_e is None or rhs_e is None:
                    scores.append(-1.0)
                    continue
                ok = sp.simplify(lhs_e - rhs_e) == 0
                scores.append(1.0 if ok else -1.0)
            except Exception:
                scores.append(-1.0)
        if not scores:
            return -1.0
        return float(sum(scores) / len(scores))

    def dense_verifier(completions, answer, **kwargs) -> Dict[str, List[float]]:
        proc, fmt, corr, total = [], [], [], []
        w_proc, w_corr, w_fmt = 0.8, 0.15, 0.05
        for resp, true_answer in zip(completions, answer):
            r_proc = verifier_process_score(resp)
            proc.append(float(r_proc))

            f = 0.0
            f += 0.05 if resp.count(rs) == 1 else -0.05
            f += 0.05 if resp.count(re_) == 1 else -0.05
            f += 0.05 if resp.count(ss) == 1 else -0.05
            f += 0.05 if resp.count(se) == 1 else -0.05
            r_fmt = max(-0.2, min(0.2, f)) / 0.2
            fmt.append(float(r_fmt))

            sol = extract_solution_str_loose(resp)
            ok = answers_equal(sol, true_answer)
            corr.append(1.0 if ok else 0.0)
            r_corr = 1.0 if ok else 0.0

            r = w_proc * r_proc + w_corr * r_corr + w_fmt * r_fmt
            r = max(-1.0, min(1.0, r))
            total.append(float(r))

        return {
            "verifier_proc": proc,
            "verifier_fmt": fmt,
            "verifier_correct": corr,
            "total_reward": total,
        }

    def split_steps_from_reasoning(resp: str) -> List[str]:
        rb = extract_reasoning_block(resp) or ""
        chunks = [s.strip() for s in re.split(r"\n\s*\n", rb) if s.strip()]
        if len(chunks) <= 1:
            chunks = [s.strip() for s in rb.splitlines() if s.strip()]
        if CFG.prm_max_steps_scored > 0:
            chunks = chunks[: int(CFG.prm_max_steps_scored)]
        return chunks

    @torch.no_grad()
    def qwen_prm_step_scores(question_text: str, steps: List[str]) -> List[float]:
        assert prm_model is not None and prm_tokenizer is not None, "PRM not loaded"
        if not steps:
            return []
        cleaned_steps = [s.strip() for s in steps if s and s.strip()]
        if not cleaned_steps:
            return []

        assistant_content = "<extra_0>".join(cleaned_steps) + "<extra_0>"
        messages = [
            {"role": "system", "content": CFG.prm_system_prompt},
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": assistant_content},
        ]
        conversation_str = prm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        input_ids = prm_tokenizer.encode(
            conversation_str,
            return_tensors="pt",
            truncation=True,
            max_length=CFG.max_seq_length,
        )
        prm_dev = next(prm_model.parameters()).device
        input_ids = input_ids.to(prm_dev)
        outputs = prm_model(input_ids=input_ids)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        if logits.shape[-1] != 2:
            raise RuntimeError(
                f"Expected scalar PRM logits with 2 labels, got shape {tuple(logits.shape)}. "
                "Use a real PRM such as Qwen/Qwen2.5-Math-PRM-7B or 72B."
            )
        probs = torch.softmax(logits, dim=-1)

        step_sep_ids = prm_tokenizer.encode("<extra_0>", add_special_tokens=False)
        if len(step_sep_ids) != 1:
            raise RuntimeError("Expected <extra_0> to map to a single token for Qwen PRM.")
        step_sep_id = step_sep_ids[0]
        token_mask = input_ids[0] == step_sep_id
        sep_probs = probs[0][token_mask]
        if sep_probs.numel() == 0:
            return []
        step_scores = sep_probs[:, 1].detach().float().cpu().tolist()
        return [float(x) for x in step_scores]

    def prm_scalar_reward(completions, answer, questions_text, **kwargs) -> Dict[str, List[float]]:
        proc, corr, total, nsteps = [], [], [], []
        alpha = float(CFG.prm_alpha)
        for resp, true_answer, qtxt in zip(completions, answer, questions_text):
            steps = split_steps_from_reasoning(resp)
            if not steps:
                step_scores = []
                r_proc = 0.0
            else:
                step_scores = qwen_prm_step_scores(qtxt, steps)
                r_proc = float(sum(step_scores) / len(step_scores)) if step_scores else 0.0
            sol = extract_solution_str_loose(resp)
            ok = answers_equal(sol, true_answer)
            r_out = (1.0 if ok else 0.0) * float(CFG.prm_outcome_scale)

            proc.append(float(r_proc))
            corr.append(float(1.0 if ok else 0.0))
            nsteps.append(float(len(step_scores)))

            if CFG.prm_include_outcome:
                r = alpha * r_proc + (1.0 - alpha) * r_out
            else:
                r = r_proc
            total.append(float(max(0.0, min(1.0, r))))

        return {
            "prm_proc": proc,
            "prm_correct": corr,
            "prm_num_scored_steps": nsteps,
            "total_reward": total,
        }

    def split_thoughts_from_reasoning(resp: str) -> List[str]:
        rb = extract_reasoning_block(resp) or ""
        if not rb.strip():
            return []
        if CFG.thought_sep_mode == "blankline":
            thoughts = [s.strip() for s in re.split(r"\n\s*\n", rb) if s.strip()]
            if len(thoughts) <= 1:
                thoughts = [s.strip() for s in rb.splitlines() if s.strip()]
        else:
            thoughts = [s.strip() for s in rb.splitlines() if s.strip()]
        if CFG.prm_max_steps_scored > 0:
            thoughts = thoughts[: int(CFG.prm_max_steps_scored)]
        return thoughts

    def tp_prm_reward(completions, answer, questions_text, **kwargs) -> Dict[str, List[float]]:
        """
        Thought-level PRM reward:
        1) segment reasoning into thoughts
        2) verifier scores each thought as correct / incorrect / unsure
        3) optionally merge contiguous same-sign thoughts
        4) apply capability-adaptive weighting based on group correctness
        5) differentiate shaping for correct vs incorrect final solutions
        6) scalarize with suffix-weighted aggregation to remain compatible with vanilla GRPO reward API
        """
        all_rows: List[Dict[str, Any]] = []
        raw_totals: List[float] = []

        # First pass: annotate each completion
        for resp, true_answer, qtxt in zip(completions, answer, questions_text):
            pred = extract_solution_str_loose(resp)
            final_ok = answers_equal(pred, true_answer)
            thoughts_txt = split_thoughts_from_reasoning(resp)

            previous: List[str] = []
            annotated: List[Dict[str, Any]] = []
            for th in thoughts_txt:
                score_raw, label = verifier.score_thought(
                    question=qtxt,
                    gold_answer=true_answer,
                    previous_thoughts=previous,
                    current_thought=th,
                    final_prediction=pred,
                    final_is_correct=final_ok,
                )
                annotated.append({
                    "text": th,
                    "label": label,
                    "score_raw": float(score_raw),
                    "num_atomic": 1,
                })
                previous.append(th)

            if CFG.tp_merge_same_sign:
                annotated = merge_same_sign_thoughts(annotated)

            raw_totals.append(1.0 if final_ok else 0.0)
            all_rows.append({
                "pred": pred,
                "final_ok": final_ok,
                "thoughts": annotated,
            })

        # Group proficiency over K samples from the same prompt
        K = int(getattr(CFG, "num_generations", 1))
        group_accs: List[float] = []
        for i in range(0, len(all_rows), K):
            chunk = all_rows[i:i+K]
            if not chunk:
                continue
            acc = sum(1.0 if r["final_ok"] else 0.0 for r in chunk) / float(len(chunk))
            group_accs.extend([acc] * len(chunk))

        totals = []
        proc_strength = []
        correct_flags = []
        nthoughts = []
        ncorrect_thoughts = []
        nincorrect_thoughts = []

        for idx, row in enumerate(all_rows):
            final_ok = bool(row["final_ok"])
            thoughts = row["thoughts"]
            group_acc = float(group_accs[idx]) if idx < len(group_accs) else 0.0

            capability_alpha = group_acc if CFG.tp_capability_adaptive else 0.5
            wrong_focus = 1.0 - capability_alpha

            thought_rewards: List[float] = []
            cnt_c = 0
            cnt_i = 0

            for th in thoughts:
                label = th["label"]
                if final_ok:
                    if label == "incorrect":
                        cnt_i += 1
                        r = float(CFG.tp_incorrect_thought_penalty) * max(0.25, capability_alpha)
                    elif label == "correct":
                        cnt_c += 1
                        r = float(CFG.tp_correct_thought_reward) * max(0.25, capability_alpha)
                    else:
                        r = float(CFG.tp_neutral_thought_reward)
                    if CFG.tp_only_penalize_error_thoughts and label == "unsure":
                        r = 0.0
                else:
                    if label == "incorrect":
                        cnt_i += 1
                        r = float(CFG.tp_incorrect_thought_penalty) * max(0.25, wrong_focus)
                    elif label == "correct":
                        cnt_c += 1
                        # Do not heavily punish locally-correct thoughts in globally-wrong traces
                        r = 0.25 * float(CFG.tp_correct_thought_reward) * max(0.10, wrong_focus)
                    else:
                        r = 0.0 if CFG.tp_leave_unmatched_incorrect_zero else float(CFG.tp_neutral_thought_reward)
                thought_rewards.append(float(r))

            # Sequence-level proxy for thought-level tail advantages
            proc_term = suffix_weighted_sum(thought_rewards)

            # final-answer shaping
            if final_ok:
                total = proc_term + float(CFG.tp_correct_path_bonus) + float(CFG.tp_outcome_bonus_scale) * group_acc
            else:
                total = proc_term + float(CFG.tp_wrong_final_penalty) - float(CFG.tp_outcome_bonus_scale) * group_acc

            totals.append(float(total))
            proc_strength.append(float(proc_term))
            correct_flags.append(1.0 if final_ok else 0.0)
            nthoughts.append(float(len(thoughts)))
            ncorrect_thoughts.append(float(cnt_c))
            nincorrect_thoughts.append(float(cnt_i))

            row["thought_rewards"] = thought_rewards
            row["proc_term"] = proc_term
            row["group_acc"] = group_acc
            row["capability_alpha"] = capability_alpha

        totals = normalize_group(totals, CFG.tp_group_std_eps)

        return {
            "prm_tp_proc": proc_strength,
            "prm_tp_correct": correct_flags,
            "prm_tp_num_thoughts": nthoughts,
            "prm_tp_num_correct_thoughts": ncorrect_thoughts,
            "prm_tp_num_incorrect_thoughts": nincorrect_thoughts,
            "total_reward": totals,
            "__tp_rows__": all_rows,
        }

    def _reward_and_account(prompts=None, completions=None, answer=None, question=None, **kw):
        K = int(getattr(CFG, "num_generations", 1))
        comps = completions or []
        prompts_list = prompts or []
        answers_batch = answer or []
        questions_batch = question or []

        if not answers_batch:
            answers_per_completion = [None] * len(comps)
        else:
            answers_per_completion = [answers_batch[j // K] for j in range(len(comps))]

        questions_expanded = (
            [questions_batch[j // K] for j in range(len(comps))]
            if questions_batch else [""] * len(comps)
        )

        pt: List[int] = []
        for p in prompts_list:
            try:
                pt.append(len(tokenizer(p, add_special_tokens=False)["input_ids"]))
            except Exception:
                pt.append(len(str(p).split()))

        mode = CFG.reward_mode.lower().strip()
        if mode == "sparse":
            components = sparse_correctness(comps, answers_per_completion)
        elif mode == "structured":
            components = structured_codeA(comps, answers_per_completion)
        elif mode == "dense":
            components = dense_bounded(comps, answers_per_completion)
        elif mode == "dense_verifier":
            components = dense_verifier(comps, answers_per_completion)
        elif mode == "prm_scalar":
            components = prm_scalar_reward(
                comps, answers_per_completion, questions_text=questions_expanded
            )
        elif mode == "prm_tp":
            components = tp_prm_reward(
                comps, answers_per_completion, questions_text=questions_expanded
            )   
        else:
            raise ValueError("reward_mode must be: sparse|structured|dense|dense_verifier|prm_scalar|prm_tp")

        # Keep TP rows out of reward logger. We no longer write them to rl_traces.jsonl.
        components.pop("__tp_rows__", None)

        total_reward = components["total_reward"]

        scale = float(CFG.reward_mode_scale.get(mode, 1.0))
        shift = float(CFG.reward_mode_shift.get(mode, 0.0))
        total_reward = [float(scale * (r + shift)) for r in total_reward]
        total_reward = [
            max(CFG.reward_clip_min, min(CFG.reward_clip_max, r))
            for r in total_reward
        ]

        if CFG.reward_standardize and len(total_reward) > 0:
            m = sum(total_reward) / float(len(total_reward))
            v = sum((r - m) ** 2 for r in total_reward) / float(len(total_reward))
            s = (v ** 0.5) if v > 1e-8 else 1.0
            total_reward = [(r - m) / s for r in total_reward]
            total_reward = [
                max(CFG.reward_clip_min, min(CFG.reward_clip_max, r))
                for r in total_reward
            ]

        components["total_reward"] = total_reward

        # Optional KL proxy computation still works, but is not written to trace files.
        if CFG.log_kl_proxy and (train_model is not None) and (ref_model is not None):
            dev = _device()
            _ = mean_logp(train_model, tokenizer, comps, device=dev)
            _ = mean_logp(ref_model, tokenizer, comps, device=dev)

        reward_logger.add_batch({
            k: v for k, v in components.items()
            if isinstance(v, list)
        })

        # IMPORTANT:
        # Keep this loop for FLOP accounting.
        # We removed only the rl_traces.jsonl writing.
        rollout_tokens = 0
        update_tokens = 0

        for j, text in enumerate(comps):
            b = j // K

            try:
                ct = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            except Exception:
                ct = len(str(text).split())

            prompt_tok = pt[b] if b < len(pt) else 0

            if CFG.reuse_prefill_across_K and (j % K) != 0:
                rollout_tokens += ct
            else:
                rollout_tokens += prompt_tok + ct

            update_tokens += prompt_tok + ct

        accountant.add(
            rollout_tokens=rollout_tokens,
            update_tokens=update_tokens,
        )

        return total_reward

    def _reward_and_accountOLD(prompts=None, completions=None, answer=None, question=None, **kw):
        K = int(getattr(CFG, "num_generations", 1))
        comps = completions or []
        prompts_list = prompts or []
        answers_batch = answer or []
        questions_batch = question or []

        if not answers_batch:
            answers_per_completion = [None] * len(comps)
        else:
            answers_per_completion = [answers_batch[j // K] for j in range(len(comps))]

        questions_expanded = (
            [questions_batch[j // K] for j in range(len(comps))]
            if questions_batch else [""] * len(comps)
        )

        pt: List[int] = []
        for p in prompts_list:
            try:
                pt.append(len(tokenizer(p, add_special_tokens=False)["input_ids"]))
            except Exception:
                pt.append(len(str(p).split()))

        mode = CFG.reward_mode.lower().strip()
        if mode == "sparse":
            components = sparse_correctness(comps, answers_per_completion)
        elif mode == "structured":
            components = structured_codeA(comps, answers_per_completion)
        elif mode == "dense":
            components = dense_bounded(comps, answers_per_completion)
        elif mode == "dense_verifier":
            components = dense_verifier(comps, answers_per_completion)
        elif mode == "prm_scalar":
            components = prm_scalar_reward(comps, answers_per_completion, questions_text=questions_expanded)
        elif mode == "prm_tp":
            components = tp_prm_reward(comps, answers_per_completion, questions_text=questions_expanded)
        else:
            raise ValueError("reward_mode must be: sparse|structured|dense|dense_verifier|prm_scalar|prm_tp")

        tp_rows = components.pop("__tp_rows__", None)
        total_reward = components["total_reward"]

        scale = float(CFG.reward_mode_scale.get(mode, 1.0))
        shift = float(CFG.reward_mode_shift.get(mode, 0.0))
        total_reward = [float(scale * (r + shift)) for r in total_reward]
        total_reward = [max(CFG.reward_clip_min, min(CFG.reward_clip_max, r)) for r in total_reward]

        if CFG.reward_standardize and len(total_reward) > 0:
            m = sum(total_reward) / float(len(total_reward))
            v = sum((r - m) ** 2 for r in total_reward) / float(len(total_reward))
            s = (v ** 0.5) if v > 1e-8 else 1.0
            total_reward = [(r - m) / s for r in total_reward]
            total_reward = [max(CFG.reward_clip_min, min(CFG.reward_clip_max, r)) for r in total_reward]

        components["total_reward"] = total_reward

        kl_new_minus_ref = None
        if CFG.log_kl_proxy and (train_model is not None) and (ref_model is not None):
            dev = _device()
            new_lp = mean_logp(train_model, tokenizer, comps, device=dev)
            ref_lp = mean_logp(ref_model, tokenizer, comps, device=dev)
            kl_new_minus_ref = [float(n - r) for n, r in zip(new_lp, ref_lp)]

        reward_logger.add_batch({k: v for k, v in components.items() if isinstance(v, list)})

        rollout_tokens = 0
        update_tokens = 0

        with open(path_in_logs(CFG.traces_filename), "a") as fout:
            for j, text in enumerate(comps):
                b = j // K
                try:
                    ct = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                except Exception:
                    ct = len(str(text).split())

                if CFG.reuse_prefill_across_K and (j % K) != 0:
                    rollout_tokens += ct
                    rollout_pt = 0
                else:
                    rollout_tokens += (pt[b] + ct)
                    rollout_pt = pt[b]

                update_tokens += (pt[b] + ct)
                update_pt = pt[b]

                rollout_flops = (accountant.c_fwd * (accountant.N + accountant.R)) * float(rollout_pt + ct)
                update_flops = (accountant.c_bwd * (accountant.N * accountant.eff_frac)) * float(update_pt + ct)

                strict_sol = extract_solution_strict(text)
                loose_sol = extract_solution_str_loose(text)
                true_ans = answers_batch[b] if b < len(answers_batch) else None

                is_correct = answers_equal(loose_sol, true_ans)
                fmt_ok = (match_format.search(text) is not None) if mode != "sparse" else (extract_sparse_answer(text) is not None)
                rb = extract_reasoning_block(text) or ""
                n_steps = len([s for s in rb.splitlines() if s.strip()])

                row = {
                    "prompt_index": int(b),
                    "completion_index": int(j),
                    "prompt_tokens": int(pt[b]) if b < len(pt) else 0,
                    "completion_tokens": int(ct),
                    "rollout_flops": float(rollout_flops),
                    "update_flops": float(update_flops),
                    "total_flops": float(rollout_flops + update_flops),
                    "reward_total": float(total_reward[j]),
                    "reward_mode": mode,
                    "metric_correct": float(1.0 if is_correct else 0.0),
                    "metric_format_ok": float(1.0 if fmt_ok else 0.0),
                    "metric_has_reasoning": float(1.0 if (extract_reasoning_block(text) is not None) else 0.0),
                    "metric_reasoning_steps": int(n_steps),
                    "metric_hit_max_length": int(ct >= CFG.max_completion_length),
                    "question": questions_batch[b] if b < len(questions_batch) else None,
                    "gold_answer_raw": true_ans,
                    "gold_answer_canon": canonical_answer(true_ans),
                    "pred_answer_raw_strict": strict_sol,
                    "pred_answer_raw_loose": loose_sol,
                    "pred_answer_canon": canonical_answer(loose_sol),
                    "raw_completion_preview": text[: CFG.debug_completion_preview_chars],
                    "raw_completion": text,
                }

                if kl_new_minus_ref is not None:
                    row["metric_logp_new_minus_ref"] = float(kl_new_minus_ref[j])

                for k, vals in components.items():
                    if k == "total_reward":
                        continue
                    if isinstance(vals, list) and j < len(vals):
                        row[f"reward_{k}"] = float(vals[j])

                if tp_rows is not None and j < len(tp_rows):
                    row["tp_prm_group_acc"] = float(tp_rows[j].get("group_acc", 0.0))
                    row["tp_prm_capability_alpha"] = float(tp_rows[j].get("capability_alpha", 0.0))
                    row["tp_prm_proc_term"] = float(tp_rows[j].get("proc_term", 0.0))
                    row["tp_prm_thoughts"] = tp_rows[j].get("thoughts", [])
                    row["tp_prm_thought_rewards"] = tp_rows[j].get("thought_rewards", [])

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        accountant.add(rollout_tokens=rollout_tokens, update_tokens=update_tokens)
        return total_reward

    return [_reward_and_account]


# ------------------- Parameter counting ----------------------

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _maybe_get(module, path):
    cur = module
    for p in path.split("."):
        cur = getattr(cur, p, None)
        if cur is None:
            return None
    return cur


# ---------------------------- Train --------------------------


def load_prm_if_needed():
    mode = CFG.reward_mode.lower().strip()

    scalar_prm_model = None
    scalar_prm_tokenizer = None
    scalar_prm_params = 0

    verifier_model_name = CFG.verifier_model_name.strip() or CFG.prm_model_name.strip()
    verifier = RealPRMVerifier(verifier_model_name, CFG.verifier_device)

    if mode == "prm_scalar":
        if not CFG.prm_model_name:
            raise RuntimeError("reward_mode=prm_scalar requires --prm_model_name")
        scalar_prm_tokenizer = AutoTokenizer.from_pretrained(
            CFG.prm_model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        scalar_prm_model = AutoModel.from_pretrained(
            CFG.prm_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if CFG.prm_device == "cuda" else None,
        ).eval()
        if CFG.prm_device == "cpu":
            scalar_prm_model.to("cpu")
        for p in scalar_prm_model.parameters():
            p.requires_grad_(False)
        scalar_prm_params = int(sum(p.numel() for p in scalar_prm_model.parameters()))

    if mode == "prm_tp" and not verifier_model_name:
        raise RuntimeError("reward_mode=prm_tp requires --verifier_model_name or --prm_model_name")

    reward_params = max(int(scalar_prm_params), int(verifier.num_params))
    return scalar_prm_model, scalar_prm_tokenizer, verifier, reward_params


def extract_solution_str_marked(text: str) -> Optional[str]:
    ss, se = CFG.solution_start, CFG.solution_end
    m = re.search(rf"{re.escape(ss)}(.+?){re.escape(se)}", text, flags=re.MULTILINE | re.DOTALL)
    return None if m is None else m.group(1).strip()


def extract_sparse_answer_validation(text: str) -> Optional[str]:
    ans_line = _extract_answer_line(text or "")
    if ans_line:
        return ans_line.strip()
    boxed = _extract_boxed_content(text or "")
    if boxed:
        return boxed.strip()
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if lines and len(lines[-1]) <= 80:
        return lines[-1]
    return None


def extract_solution_str_validation_loose(text: str) -> Optional[str]:
    if CFG.reward_mode.lower().strip() == "sparse":
        return extract_sparse_answer_validation(text)
    strict = extract_solution_str_marked(text)
    if strict is not None:
        return strict
    boxed = _extract_boxed_content(text or "")
    if boxed:
        return boxed.strip()
    ans_line = _extract_answer_line(text or "")
    if ans_line:
        return ans_line.strip()
    return None


def extract_reasoning_block_marked(text: str) -> Optional[str]:
    rs, re_ = CFG.reasoning_start, CFG.reasoning_end
    m = re.search(rf"{re.escape(rs)}(.*?){re.escape(re_)}", text, flags=re.DOTALL)
    return None if m is None else m.group(1)


def validation_format_ok(text: str) -> bool:
    if CFG.reward_mode.lower().strip() == "sparse":
        return bool(re.search(
            rf"{re.escape(CFG.sparse_answer_prefix)}\s*.*\\boxed\{{.+\}}",
            text or "",
            flags=re.DOTALL,
        ))
    rs, re_, ss, se = CFG.reasoning_start, CFG.reasoning_end, CFG.solution_start, CFG.solution_end
    pat = re.compile(
        rf"^[\s]{{0,}}{re.escape(rs)}.+?{re.escape(re_)}.*?{re.escape(ss)}(.+?){re.escape(se)}[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL,
    )
    return pat.search(text) is not None


@torch.no_grad()
def evaluate_validation_set(model, tokenizer, val_ds):
    if not CFG.run_validation or len(val_ds) == 0:
        return {}

    n_eval = len(val_ds) if CFG.validation_max_examples < 0 else min(len(val_ds), CFG.validation_max_examples)
    eval_ds = val_ds.select(range(n_eval)).with_format("python")

    try:
        FastLanguageModel.for_inference(model)
    except Exception:
        pass

    model.eval()
    device = _device()
    out_path = path_in_logs(CFG.validation_predictions_filename)

    total = 0
    correct = 0
    fmt_ok = 0
    has_reasoning = 0
    completion_tok_sum = 0.0
    hit_max_len_count = 0

    with open(out_path, "w") as fout:
        for ex in eval_ds:
            prompt = ex["prompt"]
            question = ex["question"]
            gold_answer = ex["answer"]
            example_id = int(ex["example_id"])

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=CFG.max_seq_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            prompt_len = int(enc["input_ids"].shape[1])

            gen_kwargs = dict(
                **enc,
                max_new_tokens=CFG.max_completion_length,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            if CFG.validation_temperature > 0.0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = float(CFG.validation_temperature)
                gen_kwargs["top_p"] = float(CFG.validation_top_p)
            else:
                gen_kwargs["do_sample"] = False

            out = model.generate(**gen_kwargs)
            new_ids = out[0][prompt_len:]
            completion = tokenizer.decode(new_ids, skip_special_tokens=True)

            pred_sol_strict = extract_solution_str_marked(completion)
            pred_sol_loose = extract_solution_str_validation_loose(completion)
            is_correct = answers_equal(pred_sol_loose, gold_answer)
            is_fmt_ok = validation_format_ok(completion)
            rb = extract_reasoning_block_marked(completion)
            reasoning_steps = len([s for s in (rb or "").splitlines() if s.strip()])
            hit_max_len = int(len(new_ids) >= CFG.max_completion_length)

            total += 1
            correct += int(is_correct)
            fmt_ok += int(is_fmt_ok)
            has_reasoning += int(rb is not None)
            completion_tok_sum += float(len(new_ids))
            hit_max_len_count += hit_max_len

            row = {
                "example_id": example_id,
                "question": question,
                "gold_answer_raw": gold_answer,
                "gold_answer_canon": canonical_answer(gold_answer),
                "pred_answer_raw_strict": pred_sol_strict,
                "pred_answer_raw_loose": pred_sol_loose,
                "pred_answer_canon": canonical_answer(pred_sol_loose),
                "correct": int(is_correct),
                "format_ok": int(is_fmt_ok),
                "has_reasoning": int(rb is not None),
                "reasoning_steps": int(reasoning_steps),
                "completion_tokens": int(len(new_ids)),
                "hit_max_length": hit_max_len,
                "raw_completion": completion,
            }
            if "difficulty" in ex:
                row["difficulty"] = ex["difficulty"]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "num_examples": int(total),
        "accuracy_exact": float(correct / total) if total > 0 else 0.0,
        "format_ok_rate": float(fmt_ok / total) if total > 0 else 0.0,
        "has_reasoning_rate": float(has_reasoning / total) if total > 0 else 0.0,
        "mean_completion_tokens": float(completion_tok_sum / total) if total > 0 else 0.0,
        "hit_max_length_rate": float(hit_max_len_count / total) if total > 0 else 0.0,
    }
    print("[Validation]", json.dumps(metrics, indent=2))
    return metrics


def save_split_manifest(train_ds, val_ds):
    manifest = {
        "dataset_name": CFG.dataset_name,
        "train_size": int(len(train_ds)),
        "validation_size": int(len(val_ds)),
        "validation_seed": int(CFG.validation_seed),
        "train_example_ids": [int(x) for x in train_ds["example_id"]],
        "validation_example_ids": [int(x) for x in val_ds["example_id"]],
    }
    with open(path_in_logs(CFG.split_manifest_filename), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ----------------------------- Main Train -----------------------------

def train_one_run():
    set_seed(CFG.seed)
    ensure_dirs()

    traces_path = path_in_logs(CFG.traces_filename)
    if os.path.exists(traces_path):
        os.remove(traces_path)

    print(f"[DEBUG CFG.model_name before load] {CFG.model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG.model_name,
        max_seq_length=CFG.max_seq_length,
        load_in_4bit=CFG.load_in_4bit,
        load_in_8bit=CFG.load_in_8bit,
        full_finetuning=CFG.full_finetuning,
        fast_inference=False,
        max_lora_rank=CFG.lora_r,
        gpu_memory_utilization=CFG.gpu_memory_utilization,
    )

    if CFG.use_peft:
        model = FastLanguageModel.get_peft_model(
            model,
            r=CFG.lora_r,
            target_modules=CFG.lora_target_modules,
            lora_alpha=CFG.lora_alpha,
            lora_dropout=CFG.lora_dropout,
            use_gradient_checkpointing="unsloth",
            random_state=CFG.seed,
        )

    prm_model, prm_tokenizer, verifier, reward_model_params = load_prm_if_needed()
    CFG.reward_model_params = int(reward_model_params)

    ref_model = None
    if CFG.log_kl_proxy:
        ref_name = CFG.ref_model_name.strip() or CFG.model_name
        ref_model, _ = FastLanguageModel.from_pretrained(
            model_name=ref_name,
            max_seq_length=CFG.max_seq_length,
            load_in_4bit=CFG.load_in_4bit,
            load_in_8bit=CFG.load_in_8bit,
            full_finetuning=False,
            fast_inference=False,
            gpu_memory_utilization=CFG.gpu_memory_utilization,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    for n, p in model.named_parameters():
        p.requires_grad_(("lora_" in n) or ("adapter" in n))

    total_params, trainable_params = count_params(model)

    bm = None
    for path in ["base_model.model", "model", "backbone", "transformer"]:
        bm = _maybe_get(model, path)
        if bm is not None:
            break
    backbone_params = sum(p.numel() for p in (bm.parameters() if bm else model.parameters()))
    dynamic_bwd_frac = (trainable_params / backbone_params) if CFG.use_peft else 1.0

    print(f"[Params] total={total_params:,} | trainable(LoRA)={trainable_params:,} | backbone={backbone_params:,}")
    print(f"[Reward model] params={CFG.reward_model_params:,} | mode={CFG.reward_mode}")
    print(f"[Compute] bwd_frac={dynamic_bwd_frac:.6f} | reuse_prefill_across_K={CFG.reuse_prefill_across_K}")
    print(f"[KL] beta={CFG.kl_beta}")
    print(f"[Decode] L={CFG.max_completion_length} | temperature={CFG.temperature}")

    if CFG.reward_mode == "dense_verifier" and not HAS_SYMPY:
        raise RuntimeError("dense_verifier requires sympy. Install: pip install sympy")

    accountant = FlopsAccountant(
        N=backbone_params,
        R=CFG.reward_model_params,
        c_fwd=CFG.flops_scale_forward,
        c_bwd=CFG.flops_scale_backward,
        bwd_frac=dynamic_bwd_frac,
        a_backbone=CFG.update_backbone_fraction,
    )

    train_ds, val_ds = make_train_val_datasets(tokenizer)
    #save_split_manifest(train_ds, val_ds)

    reward_logger = RewardLogger([
        "total_reward",
        "sparse_correct",
        "match_format_exactly",
        "match_format_approximately",
        "check_answer",
        "check_numbers",
        "dense_format",
        "dense_closeness",
        "dense_correct",
        "verifier_proc",
        "verifier_fmt",
        "verifier_correct",
        "prm_proc",
        "prm_correct",
        "prm_num_scored_steps",
        "prm_tp_proc",
        "prm_tp_correct",
        "prm_tp_num_thoughts",
        "prm_tp_num_correct_thoughts",
        "prm_tp_num_incorrect_thoughts",
    ])

    reward_funcs = build_rewards_and_accounting(
        tokenizer,
        accountant,
        reward_logger,
        train_model=model,
        ref_model=ref_model,
        prm_model=prm_model,
        prm_tokenizer=prm_tokenizer,
        verifier=verifier,
    )

    try:
        from vllm import SamplingParams

        stop_strings = []
        if tokenizer.eos_token:
            stop_strings.append(tokenizer.eos_token)
        if CFG.reward_mode.lower().strip() != "sparse":
            stop_strings.append(CFG.solution_end)

        vllm_sampling_params = SamplingParams(
            min_p=0.1,
            top_p=1.0,
            top_k=-1,
            seed=CFG.seed,
            stop=stop_strings,
            include_stop_str_in_output=True,
        )
    except Exception:
        vllm_sampling_params = None

    args = GRPOConfig(
        #vllm_sampling_params=vllm_sampling_params,
        temperature=CFG.temperature,
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warmup_ratio,
        lr_scheduler_type=CFG.lr_scheduler_type,
        optim=CFG.optim,
        logging_steps=CFG.logging_steps,
        per_device_train_batch_size=CFG.per_device_train_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        num_generations=CFG.num_generations,
        max_prompt_length=CFG.max_prompt_length,
        max_completion_length=CFG.max_completion_length,
        max_steps=CFG.max_steps,
        save_steps=CFG.save_steps,
        save_strategy="no",
        report_to="none",
        output_dir=CFG.output_dir,
        beta=CFG.kl_beta,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        reward_funcs=reward_funcs,
        train_dataset=train_ds,
    )

    trainer.train()
    #validation_metrics = evaluate_validation_set(model, tokenizer, val_ds)

    cfg_dict = {k: v for k, v in vars(CFG).items() if not k.startswith("__")}
    summary = {
        "cfg": cfg_dict,
        "params": {
            "total_params": total_params,
            "trainable_params_lora": trainable_params,
            "backbone_params": backbone_params,
            "reward_model_params": CFG.reward_model_params,
        },
        "compute_accounting": accountant.summary(),
        "rewards": reward_logger.summary(),
        #"validation": validation_metrics,
    }

    with open(path_in_logs(CFG.summary_filename), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Summary] {path_in_logs(CFG.summary_filename)}")
    print("[FLOPs] Totals:", json.dumps(accountant.summary()["totals"], indent=2))
    #print(f"[Debug traces] {path_in_logs(CFG.traces_filename)}")
    print(f"[Validation preds] {path_in_logs(CFG.validation_predictions_filename)}")


# ----------------------------- CLI ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="GRPO Post-Training Scaling / Reward Comparison")
    ap.add_argument("--model", type=str, default=CFG.model_name)
    ap.add_argument("--out", type=str, default=CFG.output_dir)
    ap.add_argument("--steps", type=int, default=CFG.max_steps)
    ap.add_argument("--K", type=int, default=CFG.num_generations)
    ap.add_argument("--L", type=int, default=CFG.max_completion_length)
    ap.add_argument("--grad_accum", type=int, default=CFG.gradient_accumulation_steps)
    ap.add_argument("--reuse_prefill_across_K", action="store_true")

    ap.add_argument("--dataset_name", type=str, default=CFG.dataset_name)

    ap.add_argument("--validation_size", type=int, default=CFG.validation_size)
    ap.add_argument("--validation_seed", type=int, default=CFG.validation_seed)
    ap.add_argument("--validation_max_examples", type=int, default=CFG.validation_max_examples)
    ap.add_argument("--validation_temperature", type=float, default=CFG.validation_temperature)
    ap.add_argument("--validation_top_p", type=float, default=CFG.validation_top_p)
    ap.add_argument("--no_validation", action="store_true")

    ap.add_argument(
        "--reward_mode",
        type=str,
        default=CFG.reward_mode,
        choices=["sparse", "structured", "dense", "dense_verifier", "prm_scalar", "prm_tp"],
    )
    ap.add_argument("--dense_err_scale", type=float, default=CFG.dense_err_scale)

    ap.add_argument("--verifier_strict", action="store_true")
    ap.add_argument("--verifier_friendly_prompt", action="store_true")
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--update_backbone_fraction", type=float, default=CFG.update_backbone_fraction)

    # legacy scalar PRM
    ap.add_argument("--prm_model_name", type=str, default=CFG.prm_model_name)
    ap.add_argument("--prm_device", type=str, default=CFG.prm_device)
    ap.add_argument("--prm_alpha", type=float, default=CFG.prm_alpha)
    ap.add_argument("--prm_max_steps_scored", type=int, default=CFG.prm_max_steps_scored)
    ap.add_argument("--prm_include_outcome", dest="prm_include_outcome", action="store_true")
    ap.add_argument("--no_prm_include_outcome", dest="prm_include_outcome", action="store_false")
    ap.set_defaults(prm_include_outcome=CFG.prm_include_outcome)
    ap.add_argument("--prm_outcome_scale", type=float, default=CFG.prm_outcome_scale)

    # TP-PRM / verifier
    ap.add_argument("--verifier_model_name", type=str, default=CFG.verifier_model_name)
    ap.add_argument("--verifier_device", type=str, default=CFG.verifier_device)
    ap.add_argument("--verifier_max_new_tokens", type=int, default=CFG.verifier_max_new_tokens)
    ap.add_argument("--prm_tp_pos_threshold", type=float, default=CFG.prm_tp_pos_threshold)
    ap.add_argument("--prm_tp_neg_threshold", type=float, default=CFG.prm_tp_neg_threshold)
    ap.add_argument("--thought_sep_mode", type=str, default=CFG.thought_sep_mode, choices=["blankline", "line"])
    ap.add_argument("--tp_correct_thought_reward", type=float, default=CFG.tp_correct_thought_reward)
    ap.add_argument("--tp_incorrect_thought_penalty", type=float, default=CFG.tp_incorrect_thought_penalty)
    ap.add_argument("--tp_neutral_thought_reward", type=float, default=CFG.tp_neutral_thought_reward)
    ap.add_argument("--tp_correct_path_bonus", type=float, default=CFG.tp_correct_path_bonus)
    ap.add_argument("--tp_wrong_final_penalty", type=float, default=CFG.tp_wrong_final_penalty)
    ap.add_argument("--tp_outcome_bonus_scale", type=float, default=CFG.tp_outcome_bonus_scale)
    ap.add_argument("--tp_group_std_eps", type=float, default=CFG.tp_group_std_eps)
    ap.add_argument("--no_tp_capability_adaptive", action="store_true")
    ap.add_argument("--no_tp_merge_same_sign", action="store_true")
    ap.add_argument("--no_tp_only_penalize_error_thoughts", action="store_true")
    ap.add_argument("--no_tp_leave_unmatched_incorrect_zero", action="store_true")

    ap.add_argument("--reward_standardize", action="store_true")
    ap.add_argument("--reward_clip_min", type=float, default=CFG.reward_clip_min)
    ap.add_argument("--reward_clip_max", type=float, default=CFG.reward_clip_max)

    ap.add_argument("--kl_beta", type=float, default=CFG.kl_beta)
    ap.add_argument("--temperature", type=float, default=CFG.temperature)

    ap.add_argument("--ref_model_name", type=str, default=CFG.ref_model_name)
    ap.add_argument("--log_kl_proxy", action="store_true")

    return ap.parse_args()


def apply_cli_overrides(args):
    CFG.model_name = args.model
    CFG.output_dir = args.out
    CFG.max_steps = args.steps
    CFG.num_generations = args.K
    CFG.max_completion_length = args.L
    CFG.gradient_accumulation_steps = args.grad_accum
    CFG.reuse_prefill_across_K = bool(args.reuse_prefill_across_K)

    CFG.dataset_name = args.dataset_name

    CFG.reward_mode = args.reward_mode
    CFG.dense_err_scale = args.dense_err_scale
    CFG.verifier_strict = bool(args.verifier_strict)
    CFG.verifier_friendly_prompt = bool(args.verifier_friendly_prompt)
    CFG.seed = args.seed
    CFG.update_backbone_fraction = float(args.update_backbone_fraction)

    CFG.prm_model_name = args.prm_model_name
    CFG.prm_device = args.prm_device
    CFG.prm_alpha = float(args.prm_alpha)
    CFG.prm_max_steps_scored = int(args.prm_max_steps_scored)
    CFG.prm_include_outcome = bool(args.prm_include_outcome)
    CFG.prm_outcome_scale = float(args.prm_outcome_scale)

    CFG.verifier_model_name = args.verifier_model_name
    CFG.verifier_device = args.verifier_device
    CFG.verifier_max_new_tokens = int(args.verifier_max_new_tokens)
    CFG.prm_tp_pos_threshold = float(args.prm_tp_pos_threshold)
    CFG.prm_tp_neg_threshold = float(args.prm_tp_neg_threshold)
    CFG.thought_sep_mode = args.thought_sep_mode
    CFG.tp_correct_thought_reward = float(args.tp_correct_thought_reward)
    CFG.tp_incorrect_thought_penalty = float(args.tp_incorrect_thought_penalty)
    CFG.tp_neutral_thought_reward = float(args.tp_neutral_thought_reward)
    CFG.tp_correct_path_bonus = float(args.tp_correct_path_bonus)
    CFG.tp_wrong_final_penalty = float(args.tp_wrong_final_penalty)
    CFG.tp_outcome_bonus_scale = float(args.tp_outcome_bonus_scale)
    CFG.tp_group_std_eps = float(args.tp_group_std_eps)
    CFG.tp_capability_adaptive = not bool(args.no_tp_capability_adaptive)
    CFG.tp_merge_same_sign = not bool(args.no_tp_merge_same_sign)
    CFG.tp_only_penalize_error_thoughts = not bool(args.no_tp_only_penalize_error_thoughts)
    CFG.tp_leave_unmatched_incorrect_zero = not bool(args.no_tp_leave_unmatched_incorrect_zero)

    CFG.reward_standardize = bool(args.reward_standardize)
    CFG.reward_clip_min = float(args.reward_clip_min)
    CFG.reward_clip_max = float(args.reward_clip_max)

    CFG.kl_beta = float(args.kl_beta)
    CFG.temperature = float(args.temperature)

    CFG.ref_model_name = args.ref_model_name
    CFG.log_kl_proxy = bool(args.log_kl_proxy)
    CFG.validation_size = int(args.validation_size)
    CFG.validation_seed = int(args.validation_seed)
    CFG.validation_max_examples = int(args.validation_max_examples)
    CFG.validation_temperature = float(args.validation_temperature)
    CFG.validation_top_p = float(args.validation_top_p)
    CFG.run_validation = not bool(args.no_validation)


if __name__ == "__main__":
    args = parse_args()
    apply_cli_overrides(args)
    train_one_run()

