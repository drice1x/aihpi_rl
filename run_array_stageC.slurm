#!/bin/bash -eux
#SBATCH --job-name=spot_stageC_tpprm
#SBATCH --account=aisc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=patrick.wilhelm@tu-berlin.de
#SBATCH --partition=aisc-batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --time=22:00:00
#SBATCH --constraint=ARCH:X86
#SBATCH --array=0-44%4
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --export=ALL

#SBATCH --container-image=drice1x/rl-posttrain:latest
#SBATCH --container-mounts=/sc/home/patrick.wilhelm/aihpi_rl:/workspace
#SBATCH --container-workdir=/workspace

mkdir -p /sc/home/patrick.wilhelm/aihpi_rl/logs
mkdir -p /sc/home/patrick.wilhelm/aihpi_rl/runs_spot_tpprm_judge_scaling

export HF_HOME="/sc/home/patrick.wilhelm/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

cd /workspace

bash /workspace/run_grid_task_stageC.sh \
  "$SLURM_ARRAY_TASK_ID" \
  /workspace/grid_current_stageC.txt
