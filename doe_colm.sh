#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SPOT / ICLR 2026
# Compute Allocation in RL Post-Training (Search vs Learning)
#
# Two scopes:
#   1) core_scaling
#   2) reward_extension
#
# Core idea:
#   - FIX K
#   - FIX L
#   - sweep LoRA rank r
#   - sweep training steps S inside compute bands
#   - apply mild rank-aware IsoFLOP step correction
#   - use fixed held-out validation split
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

# -------------------------
# Experiment scope
# -------------------------
# core_scaling | reward_extension | all
EXP_SCOPE="${EXP_SCOPE:-core_scaling}"

# --------------------------
# Fixed knobs
# --------------------------
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

# --------------------------
# Models
# --------------------------
CORE_MODELS=(
  "unsloth/Qwen2.5-1.5B-Instruct:1.5"
  "unsloth/Qwen2.5-3B-Instruct:3.0"
  "unsloth/Qwen2.5-7B-Instruct:7.0"
)

REWARD_MODELS=(
  "unsloth/Qwen2.5-3B-Instruct:3.0"
)

# --------------------------
# Reward modes
# --------------------------
CORE_REWARDS=("sparse")
REWARD_EXTENSION_REWARDS=("sparse" "structured" "dense" "prm")

DENSE_ERR_SCALE="${DENSE_ERR_SCALE:-5.0}"
VERIFIER_PROMPT_FLAG="--verifier_friendly_prompt"
VERIFIER_STRICT_FLAG=""

PRM_MODEL_NAME="${PRM_MODEL_NAME:-Qwen/Qwen2.5-Math-PRM-7B}"
PRM_DEVICE="${PRM_DEVICE:-cuda}"
PRM_ALPHA="${PRM_ALPHA:-0.8}"
PRM_MAX_STEPS_SCORED="${PRM_MAX_STEPS_SCORED:-64}"
PRM_INCLUDE_OUTCOME="${PRM_INCLUDE_OUTCOME:-1}"
PRM_OUTCOME_SCALE="${PRM_OUTCOME_SCALE:-1.0}"

# --------------------------
# Seeds
# --------------------------
SEEDS=(42)
# later:
# SEEDS=(42 43 44)

# --------------------------
# LoRA rank sweep
# Denser rank grid for smoother allocation/capacity curves
# --------------------------
RANKS=(8 16 32 64 128 192 256)

# --------------------------
# Compute bands
# Reference steps for 1.5B at rank=32 and alloc=a100
# --------------------------
BANDS=("C1" "C3" "C10")
declare -A BASE_STEPS_REF=(
  ["C1"]=200
  ["C3"]=600
  ["C10"]=2000
)

# --------------------------
# Allocation cells
# lower -> more rollout-heavy
# higher -> more update-heavy
# --------------------------
ALLOC_CELLS=("a40" "a70" "a100" "a140" "a180")

alloc_mult () {
  case "$1" in
    a40)  echo 0.40 ;;
    a70)  echo 0.70 ;;
    a100) echo 1.00 ;;
    a140) echo 1.40 ;;
    a180) echo 1.80 ;;
    *)    echo 1.00 ;;
  esac
}

# --------------------------
# IsoFLOP rank correction
#
# Important:
# With your current FLOP accountant and a=0.85 backbone floor,
# rank changes compute only moderately.
# So rank correction should be mild.
#
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
# Output
# --------------------------
OUT_BASE="${OUT_BASE:-runs_spot_alloc_scaling}"
mkdir -p "${OUT_BASE}"

# --------------------------
# Step scaling across model sizes
# Keep fewer steps for larger models
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

want_band ()   { [[ -z "${ONLY_BAND:-}"   || "${ONLY_BAND}"   == "$1" ]]; }
want_reward () { [[ -z "${ONLY_REWARD:-}" || "${ONLY_REWARD}" == "$1" ]]; }
want_alloc ()  { [[ -z "${ONLY_ALLOC:-}"  || "${ONLY_ALLOC}"  == "$1" ]]; }
want_model ()  { [[ -z "${ONLY_MODEL:-}"  || "$1" == *"${ONLY_MODEL}"* ]]; }
want_rank ()   { [[ -z "${ONLY_RANK:-}"   || "${ONLY_RANK}"   == "$1" ]]; }
want_scope ()  { [[ "${EXP_SCOPE}" == "all" || "${EXP_SCOPE}" == "$1" ]]; }

run_one () {
  local scope="$1"
  local band="$2"
  local model="$3"
  local size="$4"
  local rank="$5"
  local reward="$6"
  local alloc="$7"
  local seed="$8"

  local base_ref="${BASE_STEPS_REF[$band]}"
  local amult; amult="$(alloc_mult "$alloc")"
  local kmult; kmult="$(rank_mult "$rank")"

  local steps_ref_alloc_rank
  steps_ref_alloc_rank=$(${PY} - <<PY
base=float("${base_ref}")
amult=float("${amult}")
kmult=float("${kmult}")
print(int(max(1, round(base * amult * kmult))))
PY
)

  local steps
  steps="$(scale_steps_model "${steps_ref_alloc_rank}" "${size}")"

  local K="${K_FIXED}"
  local model_tag; model_tag="$(echo "$model" | sed 's|/|__|g')"
  local reuse_tag=""
  [[ "${REUSE_PREFILL}" == "1" ]] && reuse_tag="_reuseprefill"

  local outdir="${OUT_BASE}/${scope}/${band}/${reward}/${alloc}/${model_tag}/r${rank}/seed${seed}/K${K}_L${L_FIXED}_S${steps}${reuse_tag}"
  mkdir -p "${outdir}"

  if done_guard "${outdir}"; then
    echo "SKIP: ${outdir}"
    return
  fi

  local mode_flags=(--reward_mode "${reward}" --dense_err_scale "${DENSE_ERR_SCALE}")

  if [[ "${reward}" == "dense_verifier" ]]; then
    mode_flags+=(${VERIFIER_PROMPT_FLAG})
    if [[ -n "${VERIFIER_STRICT_FLAG}" ]]; then
      mode_flags+=(${VERIFIER_STRICT_FLAG})
    fi
  fi

  if [[ "${reward}" == "prm" ]]; then
    mode_flags+=(
      --prm_model_name "${PRM_MODEL_NAME}"
      --prm_device "${PRM_DEVICE}"
      --prm_alpha "${PRM_ALPHA}"
      --prm_max_steps_scored "${PRM_MAX_STEPS_SCORED}"
      --prm_outcome_scale "${PRM_OUTCOME_SCALE}"
    )
    if [[ "${PRM_INCLUDE_OUTCOME}" == "1" ]]; then
      mode_flags+=(--prm_include_outcome)
    fi
  fi

  local reuse_flag=()
  if [[ "${REUSE_PREFILL}" == "1" ]]; then
    reuse_flag+=(--reuse_prefill_across_K)
  fi

  export LORA_R="${rank}"
  export USE_WANDB="${USE_WANDB:-1}"
  export WANDB_PROJECT="${WANDB_PROJECT:-scalingrl_spot_alloc}"

  echo "------------------------------------------------------------"
  echo " scope=${scope} | band=${band} | reward=${reward} | alloc=${alloc}"
  echo " model=${model} (size=${size}) | rank=${rank} | seed=${seed}"
  echo " K=${K} | L=${L_FIXED} | steps=${steps} | reuse_prefill=${REUSE_PREFILL}"
  echo " validation_size=${VALIDATION_SIZE} | validation_max_examples=${VALIDATION_MAX_EXAMPLES}"
  if [[ "${reward}" == "prm" ]]; then
    echo " PRM=${PRM_MODEL_NAME} | prm_alpha=${PRM_ALPHA} | prm_include_outcome=${PRM_INCLUDE_OUTCOME}"
  fi
  echo " out=${outdir}"
  echo "------------------------------------------------------------"

  ${PY} "${TRAIN_SCRIPT}" \
    --model "${model}" \
    --out "${outdir}" \
    --steps "${steps}" \
    --K "${K}" \
    --L "${L_FIXED}" \
    --grad_accum "${GRAD_ACCUM}" \
    --seed "${seed}" \
    --update_backbone_fraction "${UPDATE_BACKBONE_FRACTION}" \
    --validation_size "${VALIDATION_SIZE}" \
    --validation_seed "${VALIDATION_SEED}" \
    --validation_max_examples "${VALIDATION_MAX_EXAMPLES}" \
    "${reuse_flag[@]}" \
    "${mode_flags[@]}"

  run_eval "${outdir}"
}

run_scope () {
  local scope="$1"

  local -a MODELS=()
  local -a REWARDS=()

  if [[ "${scope}" == "core_scaling" ]]; then
    MODELS=("${CORE_MODELS[@]}")
    REWARDS=("${CORE_REWARDS[@]}")
  elif [[ "${scope}" == "reward_extension" ]]; then
    MODELS=("${REWARD_MODELS[@]}")
    REWARDS=("${REWARD_EXTENSION_REWARDS[@]}")
  else
    echo "Unknown scope: ${scope}"
    exit 1
  fi

  echo
  echo "============================================================"
  echo " Running scope: ${scope}"
  echo " Models: ${MODELS[*]}"
  echo " Rewards: ${REWARDS[*]}"
  echo " Bands: ${BANDS[*]}"
  echo " Alloc cells: ${ALLOC_CELLS[*]}"
  echo " Ranks: ${RANKS[*]}"
  echo " Seeds: ${SEEDS[*]}"
  echo "============================================================"

  for band in "${BANDS[@]}"; do
    want_band "${band}" || continue

    for ms in "${MODELS[@]}"; do
      local model="${ms%%:*}"
      local size="${ms##*:}"
      want_model "${model}" || continue

      for rank in "${RANKS[@]}"; do
        want_rank "${rank}" || continue

        for reward in "${REWARDS[@]}"; do
          want_reward "${reward}" || continue

          for alloc in "${ALLOC_CELLS[@]}"; do
            want_alloc "${alloc}" || continue

            for seed in "${SEEDS[@]}"; do
              run_one "${scope}" "${band}" "${model}" "${size}" "${rank}" "${reward}" "${alloc}" "${seed}"
            done
          done
        done
      done
    done
  done
}

echo "============================================================"
echo " SPOT Compute-Allocation Runner"
echo " EXP_SCOPE=${EXP_SCOPE}"
echo " Output: ${OUT_BASE}"
echo " Fixed: K=${K_FIXED}, L=${L_FIXED}, grad_accum=${GRAD_ACCUM}"
echo " Validation: size=${VALIDATION_SIZE}, max_examples=${VALIDATION_MAX_EXAMPLES}, seed=${VALIDATION_SEED}"
echo " reuse_prefill=${REUSE_PREFILL}"
echo " update_backbone_fraction=${UPDATE_BACKBONE_FRACTION}"
echo " Filters: ONLY_BAND=${ONLY_BAND:-} ONLY_REWARD=${ONLY_REWARD:-} ONLY_ALLOC=${ONLY_ALLOC:-} ONLY_MODEL=${ONLY_MODEL:-} ONLY_RANK=${ONLY_RANK:-}"
echo "============================================================"

want_scope "core_scaling" && run_scope "core_scaling"
want_scope "reward_extension" && run_scope "reward_extension"

echo
echo "============================================================"
echo " DONE. Outputs in: ${OUT_BASE}"
echo "============================================================"