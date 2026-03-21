#!/usr/bin/env python3
# rl_posttrain_train.py
# Post-Training RL (GRPO) with LoRA, deterministic reward modes:
#   - sparse:         binary correctness (0/1)
#   - structured:     Code-A style (format + answer + numeric)
#   - dense:          dense-ish numeric shaping bounded to [-1,1]
#   - dense_verifier: process-verifier reward (Sympy-checked equalities) bounded to [-1,1]
#   - prm:            Qwen/Qwen2.5-Math-PRM-7B scalarized process reward
#
# + Token-based FLOPs accounting (rollout + update, LoRA-aware)
# + Reward normalization (scale/shift + clip + optional batch z-score)
# + Logging metrics beyond reward (correctness, format_ok, reasoning_steps, KL-proxy via logp(new)-logp(ref))

import os
import re
import json
import random
import argparse
import math
from typing import List, Dict, Any, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModel, AutoTokenizer

# Optional dependency for dense_verifier mode
try:
    import sympy as sp  # type: ignore
    HAS_SYMPY = True
except Exception:
    sp = None
    HAS_SYMPY = False

# ---- W&B ----------------------------------------------------
USE_WANDB = os.getenv("USE_WANDB", "1") == "1"
if USE_WANDB:
    import wandb
    wandb.init(project=os.getenv("WANDB_PROJECT", "scalingrl"))

# --------------------------- Config ---------------------------

class Cfg:
    # Model (keep same family across sizes)
    model_name: str = "unsloth/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 4000

    # LoRA
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = (
        "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj".split()
    )
    full_finetuning: bool = False  # backbone frozen

    # Training / budget levers
    output_dir: str = "runs/g1b_A"
    max_steps: int = 100
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    num_generations: int = 2   # K
    max_prompt_length: int = 512
    max_completion_length: int = 768
    temperature: float = 0.6
    logging_steps: int = 10
    save_steps: int = 200

    # Reward mode: sparse | structured | dense | dense_verifier | prm
    reward_mode: str = "structured"

    # --- Reward normalization ---
    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0
    reward_standardize: bool = False

    # Per-mode rescale into roughly [-1,1] BEFORE clip
    reward_mode_scale: Dict[str, float] = {
        "sparse": 1.0,           # 0/1 -> apply shift below
        "structured": 1 / 6.0,   # tune if your structured total differs
        "dense": 1.0,
        "dense_verifier": 1.0,
        "prm": 2.0,              # map [0,1] -> [-1,1] with shift=-0.5
    }
    reward_mode_shift: Dict[str, float] = {
        "sparse": -0.5,
        "structured": 0.0,
        "dense": 0.0,
        "dense_verifier": 0.0,
        "prm": -0.5,
    }

    # --- Dense params ---
    dense_err_scale: float = 5.0

    # --- PRM (Qwen/Qwen2.5-Math-PRM-7B) ---
    prm_model_name: str = ""
    prm_device: str = "cuda"         # "cuda" or "cpu"
    prm_alpha: float = 0.8           # hybrid reward: alpha*process + (1-alpha)*outcome
    prm_max_steps_scored: int = 64
    prm_include_outcome: bool = True
    prm_outcome_scale: float = 1.0
    prm_system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."

    # --- KL proxy logging ---
    log_kl_proxy: bool = False
    ref_model_name: str = ""

    # Verifier params
    verifier_strict: bool = False

    # FLOPs accounting (parameter–token estimator)
    flops_scale_forward: float = 2.0
    flops_scale_backward: float = 6.0
    reward_model_params: int = 0
    update_backbone_fraction: float = 0.85

    # Accounting switch
    reuse_prefill_across_K: bool = False

    # Data
    dataset_name: str = "polaris53k"
    dataset_config: str = None
    train_split: str = "train"

    # Markers
    reasoning_start: str = "<start_working_out>"
    reasoning_end: str = "<end_working_out>"
    solution_start: str = "<SOLUTION>"
    solution_end: str = "</SOLUTION>"

    # Prompt option for dense_verifier
    verifier_friendly_prompt: bool = False

    # Artifacts
    logs_dir: str = "logs"
    traces_filename: str = "rl_traces.jsonl"
    summary_filename: str = "summary.json"

    # System
    seed: int = 42
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    gpu_memory_utilization: float = 0.7
    
    # Validation / held-out split
    validation_size: int = 1000
    validation_seed: int = 12345
    run_validation: bool = True
    validation_max_examples: int = 1000   # <= validation_size; -1 means use all
    validation_temperature: float = 0.0   # greedy by default
    validation_top_p: float = 1.0
    validation_predictions_filename: str = "validation_predictions.jsonl"
    split_manifest_filename: str = "split_manifest.json"

CFG = Cfg()

# ---- LoRA rank override (for iso-compute sweeps) ----
if "LORA_R" in os.environ:
    CFG.lora_r = int(os.environ["LORA_R"])
    CFG.lora_alpha = 2 * CFG.lora_r
    print(f"[CFG] Overriding LoRA rank to r={CFG.lora_r}, alpha={CFG.lora_alpha}")

# ------------------------ Utilities --------------------------

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def mean_logp(model, tokenizer, texts: List[str], device: str) -> List[float]:
    """
    Average per-token log prob of each text under model (next-token prediction; ignores first token).
    """
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

# --------------------------- Data ----------------------------

CURRENT_A_KEY = "answer"

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
        "answer_parser": "math_boxed_or_lastnum",
    },
    "aime2024": {
        "hf_name": "your_org/AIME2024",  # replace if needed
        "config": None,
        "train_split": "train",
        "test_split": "test",
        "q_key": "problem",
        "a_key": "answer",
        "answer_parser": "aime_int",
    },
}

def extract_gsm8k_hash_answer(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

_num_pat = re.compile(r"-?\d+(?:\.\d+)?")

def extract_last_number(text: str) -> Optional[str]:
    nums = _num_pat.findall(text or "")
    return nums[-1] if nums else None

def _normalize_answer_text(x: Optional[str]) -> Optional[str]:
    """
    Lightweight normalization for symbolic/fractional answers.
    Keeps math content instead of collapsing everything to a number.
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None

    # Remove outer math delimiters if present
    s = s.replace("$", "").strip()

    # Normalize common spacing artifacts
    s = re.sub(r"\s+", " ", s).strip()

    # Strip a trailing period/comma/semicolon introduced by prose formatting
    s = re.sub(r"[.,;:]\s*$", "", s).strip()

    return s if s else None

def norm_ans(x: Optional[str]) -> Optional[str]:
    """
    Comparison normalization used by reward functions.

    Strategy:
    1) preserve boxed/symbolic answers when possible,
    2) canonicalize simple spacing/math wrappers,
    3) if both sides contain a number, fall back to last-number comparison.
    """
    s = _normalize_answer_text(x)
    if s is None:
        return None

    boxed = re.search(r"\\boxed\{([^}]+)\}", s)
    if boxed:
        s = boxed.group(1).strip()

    num = extract_last_number(s)
    if num is not None:
        return num

    return s

def extract_boxed_or_last_number(text: str) -> Optional[str]:
    m = re.search(r"\\boxed\{([^}]+)\}", text or "")
    if m:
        return m.group(1).strip()
    return extract_last_number(text)

def extract_polaris_answer(text: str) -> Optional[str]:
    """
    Polaris answers are already short final answers in the dataset card examples.
    They can be numeric, fractional, algebraic, or brief symbolic strings.
    Prefer boxed content if present, otherwise keep normalized raw text;
    only fall back to numeric extraction when needed.
    """
    s = _normalize_answer_text(text)
    if s is None:
        return None

    boxed = re.search(r"\\boxed\{([^}]+)\}", s)
    if boxed:
        return _normalize_answer_text(boxed.group(1))

    return s

def parse_answer(example: dict, parser: str) -> Optional[str]:
    raw = example.get(CURRENT_A_KEY, None)
    if raw is None:
        return None
    raw = str(raw)

    if parser == "gsm8k_hash":
        return extract_gsm8k_hash_answer(raw)
    if parser == "math_boxed_or_lastnum":
        return extract_boxed_or_last_number(raw)
    if parser == "aime_int":
        m = re.findall(r"-?\d+", raw)
        return m[-1].strip() if m else None
    if parser == "polaris_final":
        return extract_polaris_answer(raw)

    return extract_last_number(raw)

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
    return train, val
# -------------------- FLOPs Accountant -----------------------

class FlopsAccountant:
    """
    Rollout FLOPs: (c_fwd * (N + R)) * rollout_tokens
    Update  FLOPs: (c_bwd * (N * eff_frac)) * update_tokens
      where eff_frac = a_backbone + (1-a_backbone) * f_lora
    """
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

# ---------------- Reward helpers ----------------

def _safe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(str(x).strip().replace(",", ""))
    except Exception:
        return None

# ---------------- Rewards + Accounting ----------------

def build_rewards_and_accounting(
    tokenizer,
    accountant: FlopsAccountant,
    reward_logger: RewardLogger,
    train_model=None,
    ref_model=None,
    prm_model=None,
    prm_tokenizer=None,
):
    rs, re_, ss, se = CFG.reasoning_start, CFG.reasoning_end, CFG.solution_start, CFG.solution_end

    match_format = re.compile(
        rf"^[\s]{{0,}}{re.escape(rs)}.+?{re.escape(re_)}.*?{re.escape(ss)}(.+?){re.escape(se)}[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL,
    )
    match_numbers = re.compile(
        rf"{re.escape(ss)}.*?([-+]?\d+(?:\.\d+)?)", flags=re.MULTILINE | re.DOTALL
    )
    eq_pat = re.compile(r"^\s*(.+?)\s*=\s*(.+?)\s*$")

    def extract_solution_str(text: str) -> Optional[str]:
        m = match_format.search(text)
        return None if m is None else m.group(1).strip()

    def extract_solution_num(text: str) -> Optional[float]:
        m = match_numbers.search(text)
        return None if m is None else _safe_float(m.group(1))

    def extract_reasoning_block(text: str) -> Optional[str]:
        m = re.search(rf"{re.escape(rs)}(.*?){re.escape(re_)}", text, flags=re.DOTALL)
        return None if m is None else m.group(1)

    # ---------------- reward variants ----------------

    def sparse_correctness(completions, answer, **kwargs) -> Dict[str, List[float]]:
        corr, tot = [], []
        for resp, true_answer in zip(completions, answer):
            sol = extract_solution_str(resp)
            pred = norm_ans(sol)
            gold = norm_ans(true_answer)
            ok = (pred is not None and gold is not None and pred == gold)
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

            sol = extract_solution_str(resp)
            if sol is None:
                ans_score.append(0.0)
            else:
                ta = str(true_answer).strip()
                gs = sol.strip()
                pred = norm_ans(gs)
                gold = norm_ans(ta)

                if pred is not None and gold is not None and pred == gold:
                    ans_score.append(3.0)
                elif gs.replace(" ", "") == ta.replace(" ", ""):
                    ans_score.append(1.5)
                else:
                    g = _safe_float(gs)
                    t = _safe_float(ta)
                    if g is not None and t is not None and t != 0:
                        ratio = g / t
                        if 0.9 <= ratio <= 1.1:
                            ans_score.append(0.5)
                        elif 0.8 <= ratio <= 1.2:
                            ans_score.append(0.25)
                        else:
                            ans_score.append(-1.0)
                    else:
                        ans_score.append(-0.5)

            gnum = extract_solution_num(resp)
            tnum = _safe_float(true_answer)
            if gnum is None or tnum is None:
                num_score.append(0.0)
            else:
                num_score.append(1.5 if gnum == tnum else 0.0)

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
            ta = _safe_float(true_answer)
            gnum = extract_solution_num(resp)

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

            sol = extract_solution_str(resp)
            pred = norm_ans(sol)
            gold = norm_ans(true_answer)
            ok = (pred is not None and gold is not None and pred == gold)
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
                lhs_e = sp.sympify(lhs)
                rhs_e = sp.sympify(rhs)
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

            sol = extract_solution_str(resp)
            pred = norm_ans(sol)
            gold = norm_ans(true_answer)
            ok = (pred is not None and gold is not None and pred == gold)
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

    # ---------- Qwen Math PRM ----------

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

        enc = prm_tokenizer(
            conversation_str,
            return_tensors="pt",
            truncation=True,
            max_length=CFG.max_seq_length,
        )
        enc = {k: v.to(prm_model.device) for k, v in enc.items()}

        outputs = prm_model(**enc,use_cache=False)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        probs = torch.softmax(logits, dim=-1)

        step_sep_ids = prm_tokenizer.encode("<extra_0>", add_special_tokens=False)
        if len(step_sep_ids) != 1:
            raise RuntimeError("Expected <extra_0> to map to a single token for Qwen PRM.")
        step_sep_id = step_sep_ids[0]

        token_mask = enc["input_ids"][0] == step_sep_id
        sep_probs = probs[0][token_mask]
        if sep_probs.numel() == 0:
            return []

        step_scores = sep_probs[:, 1].detach().float().cpu().tolist()
        return [float(x) for x in step_scores]

    def prm_reward(completions, answer, questions_text, **kwargs) -> Dict[str, List[float]]:
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

            sol = extract_solution_str(resp)
            pred = norm_ans(sol)
            gold = norm_ans(true_answer)
            ok = (pred is not None and gold is not None and pred == gold)
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

    # --------------------- unified TRL hook ---------------------

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
        elif mode == "prm":
            components = prm_reward(comps, answers_per_completion, questions_text=questions_expanded)
        else:
            raise ValueError("reward_mode must be: sparse|structured|dense|dense_verifier|prm")

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

        reward_logger.add_batch(components)

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

                sol = extract_solution_str(text)
                true_ans = answers_batch[b] if b < len(answers_batch) else None
                pred = norm_ans(sol)
                gold = norm_ans(true_ans)

                is_correct = (pred is not None and gold is not None and pred == gold)
                fmt_ok = (match_format.search(text) is not None)
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
                }

                if kl_new_minus_ref is not None:
                    row["metric_logp_new_minus_ref"] = float(kl_new_minus_ref[j])

                for k, vals in components.items():
                    if k == "total_reward":
                        continue
                    if isinstance(vals, list) and j < len(vals):
                        row[f"reward_{k}"] = float(vals[j])

                fout.write(json.dumps(row) + "\n")

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
    if CFG.reward_mode.lower().strip() != "prm":
        return None, None, 0

    if not CFG.prm_model_name:
        raise RuntimeError("reward_mode=prm requires --prm_model_name")

    prm_tok = AutoTokenizer.from_pretrained(
        CFG.prm_model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    prm_model = AutoModel.from_pretrained(
        CFG.prm_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if CFG.prm_device == "cuda" else None,
    ).eval()

    if CFG.prm_device == "cpu":
        prm_model.to("cpu")

    # ---- important fix ----
    prm_model.config.use_cache = False
    if hasattr(prm_model, "generation_config") and prm_model.generation_config is not None:
        prm_model.generation_config.use_cache = False
    # -----------------------

    for p in prm_model.parameters():
        p.requires_grad_(False)

    prm_params = sum(p.numel() for p in prm_model.parameters())
    return prm_model, prm_tok, int(prm_params)

def extract_solution_str_marked(text: str) -> Optional[str]:
    rs, re_, ss, se = CFG.reasoning_start, CFG.reasoning_end, CFG.solution_start, CFG.solution_end
    m = re.search(
        rf"{re.escape(ss)}(.+?){re.escape(se)}",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return None if m is None else m.group(1).strip()

def extract_reasoning_block_marked(text: str) -> Optional[str]:
    rs, re_ = CFG.reasoning_start, CFG.reasoning_end
    m = re.search(rf"{re.escape(rs)}(.*?){re.escape(re_)}", text, flags=re.DOTALL)
    return None if m is None else m.group(1)

def validation_format_ok(text: str) -> bool:
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

    # enable inference mode for Unsloth
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

            pred_sol = extract_solution_str_marked(completion)
            pred = norm_ans(pred_sol)
            gold = norm_ans(gold_answer)

            is_correct = (pred is not None and gold is not None and pred == gold)
            is_fmt_ok = validation_format_ok(completion)
            rb = extract_reasoning_block_marked(completion)
            reasoning_steps = len([s for s in (rb or "").splitlines() if s.strip()])

            total += 1
            correct += int(is_correct)
            fmt_ok += int(is_fmt_ok)
            has_reasoning += int(rb is not None)
            completion_tok_sum += float(len(new_ids))

            row = {
                "example_id": example_id,
                "question": question,
                "gold_answer_raw": gold_answer,
                "gold_answer_norm": gold,
                "pred_answer_raw": pred_sol,
                "pred_answer_norm": pred,
                "correct": int(is_correct),
                "format_ok": int(is_fmt_ok),
                "has_reasoning": int(rb is not None),
                "reasoning_steps": int(reasoning_steps),
                "completion_tokens": int(len(new_ids)),
            }
            if "difficulty" in ex:
                row["difficulty"] = ex["difficulty"]
            fout.write(json.dumps(row) + "\n")

    metrics = {
        "num_examples": int(total),
        "accuracy_exact": float(correct / total) if total > 0 else 0.0,
        "format_ok_rate": float(fmt_ok / total) if total > 0 else 0.0,
        "has_reasoning_rate": float(has_reasoning / total) if total > 0 else 0.0,
        "mean_completion_tokens": float(completion_tok_sum / total) if total > 0 else 0.0,
        "predictions_path": out_path,
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

def train_one_run():
    set_seed(CFG.seed)
    ensure_dirs()

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

    prm_model, prm_tokenizer, prm_params = load_prm_if_needed()
    CFG.reward_model_params = int(prm_params)

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
    print(f"[PRM] params={CFG.reward_model_params:,} | mode={CFG.reward_mode}")
    print(f"[Compute] bwd_frac={dynamic_bwd_frac:.6f} | reuse_prefill_across_K={CFG.reuse_prefill_across_K}")

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

    #train_ds = make_train_dataset(tokenizer)
    train_ds, val_ds = make_train_val_datasets(tokenizer)
    split_manifest = save_split_manifest(train_ds, val_ds)

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
    ])

    reward_funcs = build_rewards_and_accounting(
        tokenizer,
        accountant,
        reward_logger,
        train_model=model,
        ref_model=ref_model,
        prm_model=prm_model,
        prm_tokenizer=prm_tokenizer,
    )

    try:
        from vllm import SamplingParams
        vllm_sampling_params = SamplingParams(
            min_p=0.1,
            top_p=1.0,
            top_k=-1,
            seed=CFG.seed,
            stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
            include_stop_str_in_output=True,
        )
    except Exception:
        vllm_sampling_params = None

    args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
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
        report_to=["wandb"] if USE_WANDB else [],
        output_dir=CFG.output_dir,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        reward_funcs=reward_funcs,
        train_dataset=train_ds,
    )

    trainer.train()
    validation_metrics = evaluate_validation_set(model, tokenizer, val_ds)


    adapter_dir = os.path.join(CFG.output_dir, "peft_adapters")
    os.makedirs(adapter_dir, exist_ok=True)
    try:
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(CFG.output_dir)
    except Exception as e:
        print(f"[WARN] Adapter save failed: {e}")

    cfg_dict = {k: v for k, v in vars(CFG).items() if not k.startswith("__")}
    summary = {
        "cfg": cfg_dict,
        "params": {
            "total_params": total_params,
            "trainable_params_lora": trainable_params,
            "backbone_params": backbone_params,
            "prm_params": CFG.reward_model_params,
        },
        "split_manifest": {
            "path": path_in_logs(CFG.split_manifest_filename),
            "train_size": len(train_ds),
            "validation_size": len(val_ds),
        },
        "compute_accounting": accountant.summary(),
        "rewards": reward_logger.summary(),
        "validation": validation_metrics,
    }

    with open(path_in_logs(CFG.summary_filename), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Summary] {path_in_logs(CFG.summary_filename)}")
    print("[FLOPs] Totals:", json.dumps(accountant.summary()["totals"], indent=2))

# ----------------------------- CLI ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="GRPO Post-Training Scaling (PEFT, FLOPs accounting) — reward modes")
    ap.add_argument("--model", type=str, default=CFG.model_name)
    ap.add_argument("--out", type=str, default=CFG.output_dir)
    ap.add_argument("--steps", type=int, default=CFG.max_steps)
    ap.add_argument("--K", type=int, default=CFG.num_generations)
    ap.add_argument("--L", type=int, default=CFG.max_completion_length)
    ap.add_argument("--grad_accum", type=int, default=CFG.gradient_accumulation_steps)
    ap.add_argument("--reuse_prefill_across_K", action="store_true")
    # Validation
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
        choices=["sparse", "structured", "dense", "dense_verifier", "prm"],
    )
    ap.add_argument("--dense_err_scale", type=float, default=CFG.dense_err_scale)

    ap.add_argument("--verifier_strict", action="store_true")
    ap.add_argument("--verifier_friendly_prompt", action="store_true")
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--update_backbone_fraction", type=float, default=CFG.update_backbone_fraction)

    # PRM
    ap.add_argument("--prm_model_name", type=str, default=CFG.prm_model_name)
    ap.add_argument("--prm_device", type=str, default=CFG.prm_device)
    ap.add_argument("--prm_alpha", type=float, default=CFG.prm_alpha)
    ap.add_argument("--prm_max_steps_scored", type=int, default=CFG.prm_max_steps_scored)
    ap.add_argument("--prm_include_outcome", action="store_true")
    ap.add_argument("--prm_outcome_scale", type=float, default=CFG.prm_outcome_scale)

    # Reward normalization
    ap.add_argument("--reward_standardize", action="store_true")
    ap.add_argument("--reward_clip_min", type=float, default=CFG.reward_clip_min)
    ap.add_argument("--reward_clip_max", type=float, default=CFG.reward_clip_max)

    # KL proxy
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

    CFG.reward_standardize = bool(args.reward_standardize)
    CFG.reward_clip_min = float(args.reward_clip_min)
    CFG.reward_clip_max = float(args.reward_clip_max)

    CFG.ref_model_name = args.ref_model_name
    CFG.log_kl_proxy = bool(args.log_kl_proxy)
    CFG.validation_size = int(args.validation_size)
    CFG.validation_seed = int(args.validation_seed)
    CFG.validation_max_examples = int(args.validation_max_examples)
    CFG.validation_temperature = float(args.validation_temperature)
    CFG.validation_top_p = float(args.validation_top_p)
    CFG.run_validation = not bool(args.no_validation)

# ----------------------------- Main --------------------------

if __name__ == "__main__":
    args = parse_args()
    apply_cli_overrides(args)
    train_one_run()