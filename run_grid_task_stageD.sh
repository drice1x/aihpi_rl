#!/bin/bash
set -euo pipefail

TASK_ID="$1"
GRID_FILE="$2"

if [ -d /workspace ]; then
  cd /workspace
else
  cd ~/aihpi_rl
fi

mapfile -t CONFIGS < "$GRID_FILE"
CFG="${CONFIGS[$TASK_ID]}"

read -r MODEL SIZE BAND REWARD PRM ALLOC LORA_RANK SEED SCOPE <<< "$CFG"

K_FIXED=2
L_FIXED=2048
GRAD_ACCUM=8

VALIDATION_SIZE="${VALIDATION_SIZE:-1000}"
VALIDATION_MAX_EXAMPLES="${VALIDATION_MAX_EXAMPLES:-1000}"
VALIDATION_SEED="${VALIDATION_SEED:-12345}"

UPDATE_BACKBONE_FRACTION="${UPDATE_BACKBONE_FRACTION:-0.85}"
BASE_STEPS_REF_C1="${BASE_STEPS_REF_C1:-200}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-rl_posttrain_train_real_prm.py}"

VERIFIER_DEVICE="${VERIFIER_DEVICE:-cuda}"
PRM_DEVICE="${PRM_DEVICE:-cuda}"

TP_POS_THRESHOLD="${TP_POS_THRESHOLD:-0.75}"
TP_NEG_THRESHOLD="${TP_NEG_THRESHOLD:-0.25}"
PRM_ALPHA="${PRM_ALPHA:-0.8}"
PRM_OUTCOME_SCALE="${PRM_OUTCOME_SCALE:-1.0}"
PRM_MAX_STEPS_SCORED="${PRM_MAX_STEPS_SCORED:-64}"

prm_model () {
  case "$1" in
    prm7)  echo "Qwen/Qwen2.5-Math-PRM-7B" ;;
    #prm72) echo "Qwen/Qwen2.5-Math-PRM-72B" ;;
    *)     echo "Qwen/Qwen2.5-Math-PRM-7B" ;;
  esac
}

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

rank_mult () {
  case "$1" in
    16)  echo 1.02 ;;
    64)  echo 0.98 ;;
    256) echo 0.91 ;;
    *)   echo 1.00 ;;
  esac
}

scale_steps_model () {
  local steps_ref="$1"
  local model_size="$2"
  python - <<PY
steps_ref=float("${steps_ref}")
size=float("${model_size}")
ref=1.5
print(int(max(1, round(steps_ref * ref / size))))
PY
}

AMULT="$(alloc_mult "$ALLOC")"
KMULT="$(rank_mult "$LORA_RANK")"
PRM_MODEL_NAME="$(prm_model "$PRM")"

STEPS_REF_ALLOC_RANK=$(python - <<PY
base=float("${BASE_STEPS_REF_C1}")
amult=float("${AMULT}")
kmult=float("${KMULT}")
print(int(max(1, round(base * amult * kmult))))
PY
)

STEPS="$(scale_steps_model "${STEPS_REF_ALLOC_RANK}" "${SIZE}")"

MODEL_TAG="$(echo "$MODEL" | sed 's|/|__|g')"
PRM_TAG="$(echo "$PRM" | sed 's|/|__|g')"

OUTDIR="runs_spot_real_tpprm/${SCOPE}/${BAND}/${REWARD}/${PRM_TAG}/${ALLOC}/${MODEL_TAG}/r${LORA_RANK}/seed${SEED}/K${K_FIXED}_L${L_FIXED}_S${STEPS}"

mkdir -p "$OUTDIR"

if [[ -f "${OUTDIR}/logs/summary.json" ]]; then
  echo "[SKIP] ${OUTDIR}"
  exit 0
fi

export LORA_R="${LORA_RANK}"

echo "============================================================"
echo "TASK_ID=${TASK_ID}"
echo "CFG=${CFG}"
echo "MODEL=${MODEL}"
echo "PRM=${PRM}"
echo "PRM_MODEL_NAME=${PRM_MODEL_NAME}"
echo "ALLOC=${ALLOC}"
echo "LORA_RANK=${LORA_RANK}"
echo "REWARD=${REWARD}"
echo "STEPS=${STEPS}"
echo "OUTDIR=${OUTDIR}"
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "============================================================"

python "${TRAIN_SCRIPT}" \
  --model "${MODEL}" \
  --out "${OUTDIR}" \
  --steps "${STEPS}" \
  --K "${K_FIXED}" \
  --L "${L_FIXED}" \
  --grad_accum "${GRAD_ACCUM}" \
  --seed "${SEED}" \
  --reward_mode "${REWARD}" \
  --prm_model_name "${PRM_MODEL_NAME}" \
  --prm_device "${PRM_DEVICE}" \
  --verifier_model_name "${PRM_MODEL_NAME}" \
  --verifier_device "${VERIFIER_DEVICE}" \
  --prm_alpha "${PRM_ALPHA}" \
  --prm_outcome_scale "${PRM_OUTCOME_SCALE}" \
  --prm_max_steps_scored "${PRM_MAX_STEPS_SCORED}" \
  --prm_tp_pos_threshold "${TP_POS_THRESHOLD}" \
  --prm_tp_neg_threshold "${TP_NEG_THRESHOLD}" \
  --update_backbone_fraction "${UPDATE_BACKBONE_FRACTION}" \
  --validation_size "${VALIDATION_SIZE}" \
  --validation_seed "${VALIDATION_SEED}" \
  --validation_max_examples "${VALIDATION_MAX_EXAMPLES}"
