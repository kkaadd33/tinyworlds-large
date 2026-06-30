#!/bin/bash
#SBATCH --job-name=dync8cor
#SBATCH --partition=ugrip
#SBATCH -q prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --time=18:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tinyworlds
cd ~/ugrip/tinyworlds-large
export WANDB_MODE=online PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/ugrip/tinyworlds-large:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NG_RUN_ROOT_DIR="results/large_zelda_ctx8_corrupt"
export WANDB_RUN_GROUP="large_zelda_ctx8_corrupt"
export DYN_CTX_CORRUPT_P="0.25"     # corrupt 25% of unmasked context latents
export DYN_CTX_CORRUPT_SIGMA="1.0"
echo "ctx8 dynamics FINE-TUNE with context-token corruption (p=0.25), warm-start from ctx8 step_29500"
nvidia-smi -L
torchrun --standalone --nproc_per_node=4 scripts/train_dynamics.py \
  --config configs/dynamics_large_ctx8_corrupt.yaml --training_config configs/training_ddp_ctx8.yaml
