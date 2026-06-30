#!/bin/bash
#SBATCH --job-name=actl8
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
export NG_RUN_ROOT_DIR="results/large_zelda_ctx8act"
export WANDB_RUN_GROUP="large_zelda_ctx8"
export LAM_ENTROPY_LAMBDA="0.1"
export LAM_VAR_LAMBDA="1.0"
echo "LARGE action CONTEXT=8 (512/2048/8, batch8x4x4acc, joint-entropy + var_lambda=1)"
nvidia-smi -L
torchrun --standalone --nproc_per_node=4 scripts/train_latent_actions.py \
  --config configs/latent_actions_large_ctx8.yaml --training_config configs/training_ddp_ctx8.yaml
