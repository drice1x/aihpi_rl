#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SPOT / ICLR 2026
# Pilot-only runner for compute-allocation plane sanity check
#
# Goal:
#   Verify IsoFLOP density and allocation trends before full sweep.
#
# Fixed:
#   - model = Qwen2.5-3B-Instruct
#   - band = C1
#   - alloc cells = {a40, a100, a180}
#   - ranks = {8,16,32,64,128,192,256}
#   - reward = sparse
#   - seed = 42
# ============================================================

PY="${PY:-python3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-rl_posttrain_train.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval_rl_run.py}"

# -------------------------
# HF / network robustness
# -------------------------
export HF_HUB_READ_TIMEOUT="${HF_HUB_READ_TIMEOUT:-120}"
export HF_HUB_CONNECT_TIMEOUT="${HF_HUB_CONNECT_TIMEOUT:-60}"
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# --------------------------
# Fixed pilot knobs
# --------------------------
MODEL="${MODEL:-unsloth/Qwen2.5-3B-Instruct}"
MODEL_SIZE="${MODEL_SIZE:-3.0}"

L_FIXED="${L_FIXED:-2048}"
K_FIXED="${K_FIXED:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

VALIDATION_SIZE="${VALIDATION_SIZE:-1000}"
VALIDATION_MAX_EXAMPLES="${VALIDATION_MAX_EXAMPLES:-1000}"
VALIDATION_SEED="${VALIDATION_SEED:-12345}"

RUN_EXTERNAL_EVAL="${RUN_EXTERNAL_EVAL:-0}"
N_EVAL="${N_EVAL:-500}"
BATCH_EVAL="${BATCH_EVAL:-4}"

REUSE_PREFILL="${REUSE_PREFILL:-0}"   # 0/1
UPDATE_BACKBONE_FRACTION="${UPDATE_BACKBONE_FRACTION:-0.85}"

REWARD_MODE="${REWARD_MODE:-sparse}"
SEED="${SEED:-42}"

# --------------------------
# Pilot sweep dimensions
# --------------------------
BAND="C1"
RANKS=(8 16 128)
ALLOC_CELLS=("a40" "a100" "a180")

# Reference steps for 1.5B at rank=32 and alloc=a100
BASE_STEPS_REF_C1="${BASE_STEPS_REF_C1:-200}"

# --------------------------
# Output
# --------------------------
OUT_BASE="${OUT_BASE:-runs_spot_alloc_pilot}"
mkdir -p "${OUT_BASE}"

# --------------------------
# Allocation multipliers
# --------------------------
alloc_mult () {
  case "$1" in
    a40)  echo 0.40 ;;
    a100) echo 1.00 ;;
    a180) echo 1.80 ;;
    *)    echo 1.00 ;;
  esac
}

# --------------------------
# Mild IsoFLOP rank correction
# Baseline = rank 32
# --------------------------
rank_mult () {
  case "$1" in
    8)   echo 1.04 ;;
    16)  echo 1.02 ;;
    32)  echo 1.00 ;;
    64)  echo 0.98 ;;
    128) echo 0.95 ;;
    192) echo 0.93 ;;
    256) echo 0.91 ;;
    *)   echo 1.00 ;;
  esac
}

# --------------------------
# Cross-model step scaling
# Ref model = 1.5B
# --------------------------
scale_steps_model () {
  local steps_ref="$1"
  local model_size="$2"
  ${PY} - <<PY
steps_ref=float("${steps_ref}")
size=float("${model_size}")
ref=1.5
print(int(max(1, round(steps_ref * ref / size))))
PY
}

# --------------------------
# Helpers
# --------------------------
done_guard () {
  local outdir="$1"
  [[ -f "${outdir}/logs/summary.json" ]]
}

run_eval () {
  local outdir="$1"
  if [[ "${RUN_EXTERNAL_EVAL}" != "1" ]]; then
    return
  fi
  if [[ -f "${outdir}/logs/eval.json" ]]; then
    echo "  [eval] SKIP (exists): ${outdir}/logs/eval.json"
    return
  fi
  ${PY} "${EVAL_SCRIPT}" --run_dir "${outdir}" --n_eval "${N_EVAL}" --batch_size "${BATCH_EVAL}"
}

run_one () {
  local rank="$1"
  local alloc="$2"

  local amult; amult="$(alloc_mult "$alloc")"
  local kmult; kmult="$(rank_mult "$rank")"

  local steps_ref_alloc_rank
  steps_ref_alloc_rank=$(${PY} - <<PY
base=float("${BASE_STEPS_REF_C1}")
amult=float("${amult}")
kmult=float("${kmult}")
print(int(max(1, round(base * amult * kmult))))
PY
)

  local steps
  steps="$(scale_steps_model "${steps_ref_alloc_rank}" "${MODEL_SIZE}")"

  local K="${K_FIXED}"
  local model_tag; model_tag="$(echo "$MODEL" | sed 's|/|__|g')"
  local reuse_tag=""
  [[ "${REUSE_PREFILL}" == "1" ]] && reuse_tag="_reuseprefill"

  local outdir="${OUT_BASE}/${BAND}/${REWARD_MODE}/${alloc}/${model_tag}/r${rank}/seed${SEED}/K${K}_L${L_FIXED}_S${steps}${reuse_tag}"
  mkdir -p "${outdir}"

  if done_guard "${outdir}"; then
    echo "SKIP: ${outdir}"
    return
  fi

  local mode_flags=(--reward_mode "${REWARD_MODE}")

  local reuse_flag=()
  if [[ "${REUSE_PREFILL}" == "1" ]]; then
    reuse_flag+=(--reuse_prefill_across_K)
  fi

  export LORA_R="${rank}"
  export USE_WANDB="${USE_WANDB:-1}"
  export WANDB_PROJECT="${WANDB_PROJECT:-scalingrl_spot_alloc_pilot}"

  echo "------------------------------------------------------------"
  echo " PILOT | band=${BAND} | reward=${REWARD_MODE} | alloc=${alloc}"
  echo " model=${MODEL} (size=${MODEL_SIZE}) | rank=${rank} | seed=${SEED}"
  echo " K=${K} | L=${L_FIXED} | steps=${steps} | reuse_prefill=${REUSE_PREFILL}"
  echo " validation_size=${VALIDATION_SIZE} | validation_max_examples=${VALIDATION_MAX_EXAMPLES}"
  echo " out=${outdir}"
  echo "------------------------------------------------------------"

  ${PY} "${TRAIN_SCRIPT}" \
    --model "${MODEL}" \
    --out "${outdir}" \
    --steps "${steps}" \
    --K "${K}" \
    --L "${L_FIXED}" \
    --grad_accum "${GRAD_ACCUM}" \
    --seed "${SEED}" \
    --update_backbone_fraction "${UPDATE_BACKBONE_FRACTION}" \
    --validation_size "${VALIDATION_SIZE}" \
    --validation_seed "${VALIDATION_SEED}" \
    --validation_max_examples "${VALIDATION_MAX_EXAMPLES}" \
    "${reuse_flag[@]}" \
    "${mode_flags[@]}"

  run_eval "${outdir}"
}

echo "============================================================"
echo " SPOT Pilot Allocation Runner"
echo " Model: ${MODEL} (size=${MODEL_SIZE})"
echo " Band: ${BAND}"
echo " Reward: ${REWARD_MODE}"
echo " Alloc cells: ${ALLOC_CELLS[*]}"
echo " Ranks: ${RANKS[*]}"
echo " Seed: ${SEED}"
echo " Fixed: K=${K_FIXED}, L=${L_FIXED}, grad_accum=${GRAD_ACCUM}"
echo " Validation: size=${VALIDATION_SIZE}, max_examples=${VALIDATION_MAX_EXAMPLES}, seed=${VALIDATION_SEED}"
echo " reuse_prefill=${REUSE_PREFILL}"
echo " update_backbone_fraction=${UPDATE_BACKBONE_FRACTION}"
echo " Output: ${OUT_BASE}"
echo "============================================================"

for rank in "${RANKS[@]}"; do
  for alloc in "${ALLOC_CELLS[@]}"; do
    run_one "${rank}" "${alloc}"
  done
done

echo
echo "============================================================"
echo " PILOT DONE. Outputs in: ${OUT_BASE}"
echo "============================================================"