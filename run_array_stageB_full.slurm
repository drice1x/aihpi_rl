#!/bin/bash -eux
#SBATCH --job-name=spot_stageB_full
#SBATCH --account=aisc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=patrick.wilhelm@tu-berlin.de
#SBATCH --partition=aisc-batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --constraint=ARCH:X86
#SBATCH --array=0-59%4
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --export=ALL

#SBATCH --container-image=drice1x/rl-posttrain:latest
#SBATCH --container-mounts=/sc/home/$USER/aihpi_rl:/workspace
#SBATCH --container-workdir=/workspace

mkdir -p /sc/home/$USER/aihpi_rl/logs
mkdir -p /sc/home/$USER/aihpi_rl/runs_spot_reward_extension

export HF_HOME="/tmp/hf_cache_${USER}"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

cd /workspace

bash /workspace/run_grid_task_stageB_full.sh \
  "$SLURM_ARRAY_TASK_ID" \
  /workspace/grid_current_stageB.txt
