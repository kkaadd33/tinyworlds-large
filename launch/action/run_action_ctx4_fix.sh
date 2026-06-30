#!/bin/bash
#SBATCH --job-name=actfix4
#SBATCH --partition=ugrip
#SBATCH -q prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tinyworlds
cd ~/ugrip/tinyworlds-large
export WANDB_MODE=online PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/ugrip/tinyworlds-large:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NG_RUN_ROOT_DIR="results/large_zelda_ctx4_fix"
export WANDB_RUN_GROUP="large_zelda_ctx4_fix"
export LAM_KEEP_RATE="1.0"        # FIX #1: decoder sees the real previous frame
export LAM_PREDICT_DELTA="1"      # FIX #2: predict the residual/delta (closes the copy shortcut)
export LAM_ENTROPY_LAMBDA="0.3"   # a touch stronger anti-collapse (delta signal is small)
export LAM_VAR_LAMBDA="1.0"
export LAM_MOTION_LAMBDA="0.0"
echo "ACTION ctx4 (STABLE config) WITH keep_rate + delta fixes -- validate the character moves"
nvidia-smi -L
torchrun --standalone --nproc_per_node=4 scripts/train_latent_actions.py \
  --config configs/latent_actions_large.yaml --training_config configs/training_ddp.yaml
