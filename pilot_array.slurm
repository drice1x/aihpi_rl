#!/usr/bin/env bash
#SBATCH --job-name=spot_pilot
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-20

set -euo pipefail

cd /path/to/your/project

source ~/miniconda3/etc/profile.d/conda.sh
conda activate yourenv

mkdir -p logs

mapfile -t CONFIGS < configs_pilot.txt
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

read -r RANK ALLOC <<< "$CFG"

MODEL="unsloth/Qwen2.5-3B-Instruct"
MODEL_SIZE="3.0"
BAND="C1"
REWARD="sparse"
SEED="42"
K="2"
L="2048"
VALIDATION_SIZE="1000"
VALIDATION_MAX_EXAMPLES="1000"
VALIDATION_SEED="12345"
BASE_STEPS_REF_C1="200"

alloc_mult () {
  case "$1" in
    a40)  echo 0.40 ;;
    a100) echo 1.00 ;;
    a180) echo 1.80 ;;
    *)    echo 1.00 ;;
  esac
}

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

AMULT=$(alloc_mult "$ALLOC")
KMULT=$(rank_mult "$RANK")

STEPS_REF_ALLOC_RANK=$(python - <<PY
base=float("${BASE_STEPS_REF_C1}")
amult=float("${AMULT}")
kmult=float("${KMULT}")
print(int(max(1, round(base * amult * kmult))))
PY
)

STEPS=$(python - <<PY
steps_ref=float("${STEPS_REF_ALLOC_RANK}")
size=float("${MODEL_SIZE}")
ref=1.5
print(int(max(1, round(steps_ref * ref / size))))
PY
)

export LORA_R="$RANK"
export USE_WANDB=1
export WANDB_PROJECT="scalingrl_spot_alloc_pilot"

OUTDIR="runs_spot_alloc_pilot/${BAND}/${REWARD}/${ALLOC}/qwen2p5_3b/r${RANK}/seed${SEED}/K${K}_L${L}_S${STEPS}"

python rl_posttrain_train.py \
  --model "$MODEL" \
  --out "$OUTDIR" \
  --steps "$STEPS" \
  --K "$K" \
  --L "$L" \
  --seed "$SEED" \
  --reward_mode "$REWARD" \
  --validation_size "$VALIDATION_SIZE" \
  --validation_seed "$VALIDATION_SEED" \
  --validation_max_examples "$VALIDATION_MAX_EXAMPLES"