# eval_rl_run.py
# Evaluate a GRPO+LoRA run on GSM8K and store accuracy + length stats.

import os
import json
import argparse
import random
import re
from typing import List, Dict, Any, Optional

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import PeftModel

# ---------------------- Utils & Config ------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_summary(run_dir: str) -> Dict[str, Any]:
    summary_path = os.path.join(run_dir, "logs", "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary.json not found at {summary_path}. "
                                f"Run training first.")
    with open(summary_path, "r") as f:
        return json.load(f)

def extract_hash_answer(text: str) -> Optional[str]:
    # Same as in training script
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# --------------------- Model Loading -------------------------

def load_finetuned_model(run_dir: str, device: str = "cuda"):

    summary = load_summary(run_dir)
    cfg = summary["cfg"]

    model_name = cfg["model_name"]
    max_seq_length = cfg.get("max_seq_length", 1024)
    load_in_4bit = cfg.get("load_in_4bit", False)
    load_in_8bit = cfg.get("load_in_8bit", False)
    gpu_memory_utilization = cfg.get("gpu_memory_utilization", 0.7)

    adapter_dir = os.path.join(run_dir, "peft_adapters")
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    print(f"[Eval] Loading base model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        load_in_8bit = load_in_8bit,
        fast_inference = False,
        max_lora_rank = cfg.get("lora_r", 16),
        gpu_memory_utilization = gpu_memory_utilization,
    )

    print(f"[Eval] Loading LoRA adapters from: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)

    # Enable fast inference
    FastLanguageModel.for_inference(model)

    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")
    else:
        device = "cpu"
        model.to("cpu")

    print(f"[Eval] Model device: {device}")
    return model, tokenizer, cfg

# ---------------------- Data & Prompts -----------------------

def make_eval_dataset(tokenizer, cfg: Dict[str, Any], n_eval: int):
    dataset_name = cfg.get("dataset_name", "openai/gsm8k")
    dataset_config = cfg.get("dataset_config", "main")

    print(f"[Eval] Loading dataset: {dataset_name} / {dataset_config} / test")
    ds = load_dataset(dataset_name, dataset_config, split="test")

    reasoning_start = cfg.get("reasoning_start", "<start_working_out>")
    reasoning_end   = cfg.get("reasoning_end", "<end_working_out>")
    solution_start  = cfg.get("solution_start", "<SOLUTION>")
    solution_end    = cfg.get("solution_end", "</SOLUTION>")

    # --- FIX: match training prompt distribution ---
    verifier_friendly = bool(cfg.get("verifier_friendly_prompt", False))
    if verifier_friendly:
        extra = (
            "In the working out, write key calculations as separate lines of the form "
            "'expression = value' so they can be automatically verified.\n"
        )
    else:
        extra = ""

    system_prompt = (
        "You are given a problem. Think about the problem and provide your working out. "
        + extra +
        f"Place it between {reasoning_start} and {reasoning_end}. "
        f"Then, provide your solution between {solution_start}{solution_end}"
    )

    max_prompt_length = cfg.get("max_prompt_length", 256)

    def map_ex(example):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": example["question"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        return {
            "prompt": prompt_text,
            "answer": extract_hash_answer(example["answer"]),
        }

    eval_ds = ds.map(map_ex, remove_columns=ds.column_names)
    eval_ds = eval_ds.filter(lambda r: r["answer"] is not None).with_format("python")

    if n_eval is not None and n_eval > 0:
        eval_ds = eval_ds.select(range(min(n_eval, len(eval_ds))))

    print(f"[Eval] Using {len(eval_ds)} examples for evaluation.")
    return eval_ds

# -------------------- Answer Extraction ----------------------

def build_solution_extractors(cfg: Dict[str, Any]):
    """
    Extract final numeric answer.

    Priority:
      1. Last number inside <SOLUTION>...</SOLUTION>, if present.
      2. Otherwise last number in the whole completion.
    """
    solution_start = cfg.get("solution_start", "<SOLUTION>")
    solution_end   = cfg.get("solution_end", "</SOLUTION>")

    sol_pattern = re.compile(
        rf"{re.escape(solution_start)}(.*?){re.escape(solution_end)}",
        flags=re.MULTILINE | re.DOTALL,
    )

    num_pattern = re.compile(r"-?\d+(?:\.\d+)?")

    def extract_pred_answer(text: str) -> Optional[str]:
        # 1) Focus on <SOLUTION>...</SOLUTION> if present
        m = sol_pattern.search(text)
        if m is not None:
            region = m.group(1)
        else:
            region = text

        # 2) Last numeric token in that region
        nums = num_pattern.findall(region)
        if nums:
            return nums[-1].strip()

        return None

    return extract_pred_answer

# ----------------------- Evaluation --------------------------

def evaluate(
    model,
    tokenizer,
    cfg: Dict[str, Any],
    eval_ds,
    device: str = "cuda",
    max_new_tokens: int = 512,
    batch_size: int = 4,
) -> Dict[str, Any]:
    model.eval()

    extract_pred_answer = build_solution_extractors(cfg)

    n = len(eval_ds)
    n_correct = 0
    lengths: List[int] = []

    # For optional debugging
    examples_logged = 0
    max_debug = 5

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = eval_ds[start : start + batch_size]  # this is a dict of lists
            prompts = batch["prompt"]
            true_answers = batch["answer"]

            # Tokenize
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.get("max_prompt_length", 256),
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate (greedy decoding)
            gen_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode completions only
            for i, prompt in enumerate(prompts):
                # real prompt length = number of non-pad tokens
                input_len = int(inputs["attention_mask"][i].sum().item())
                out_ids = gen_out[i][input_len:]  # strip prompt tokens
                completion = tokenizer.decode(out_ids, skip_special_tokens=False)

                # Token length for the completion
                length_tokens = len(
                    tokenizer(
                        completion,
                        add_special_tokens=False,
                    )["input_ids"]
                )
                lengths.append(length_tokens)

                # Extract predicted answer & compare
                pred = extract_pred_answer(completion)
                true_ans = true_answers[i]

                correct = False
                if pred is not None and true_ans is not None:
                    # Try exact string compare and numeric compare
                    if pred.strip() == true_ans.strip():
                        correct = True
                    else:
                        try:
                            p_val = float(pred)
                            t_val = float(true_ans)
                            correct = (p_val == t_val)
                        except Exception:
                            correct = False

                if correct:
                    n_correct += 1

                # Debug print a few
                if examples_logged < max_debug:
                    print("----- EXAMPLE -----")
                    print("Prompt:", prompts[i])
                    print("Completion:", completion)
                    print("True answer:", true_ans)
                    print("Predicted:", pred)
                    print("Correct:", correct)
                    examples_logged += 1

    accuracy = n_correct / n if n > 0 else 0.0
    if len(lengths) > 0:
        import numpy as np
        mean_len = float(np.mean(lengths))
        median_len = float(np.median(lengths))
        std_len = float(np.std(lengths))
    else:
        mean_len = median_len = std_len = 0.0

    results = {
        "n_eval_examples": n,
        "n_correct": int(n_correct),
        "gsm8k_accuracy": float(accuracy),
        "mean_completion_tokens": mean_len,
        "median_completion_tokens": median_len,
        "std_completion_tokens": std_len,
    }

    print("[Eval] Results:", json.dumps(results, indent=2))
    return results


# -------------------------- Main -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate GRPO+LoRA run on GSM8K.")
    ap.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory (same as --out used during training).",
    )
    ap.add_argument(
        "--n_eval",
        type=int,
        default=500,
        help="Number of test examples to evaluate on (from GSM8K test).",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens to generate for each completion.",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation generation.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    run_dir = args.run_dir
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, cfg = load_finetuned_model(run_dir, device=device)
    eval_ds = make_eval_dataset(tokenizer, cfg, n_eval=args.n_eval)

    results = evaluate(
        model,
        tokenizer,
        cfg,
        eval_ds,
        device=device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    # Save eval results next to summary.json
    eval_path = os.path.join(run_dir, "logs", "eval.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] Saved results to: {eval_path}")

    # Optional cleanup (nice for long sweeps, harmless otherwise)
    try:
        del model
        del tokenizer
    except NameError:
        pass
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
