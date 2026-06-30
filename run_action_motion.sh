#!/bin/bash
#SBATCH --job-name=actmot
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
export NG_RUN_ROOT_DIR="results/large_zelda_motion"
export WANDB_RUN_GROUP="large_zelda_motion"
export LAM_ENTROPY_LAMBDA="0.5"   # raised 0.1->0.5 so motion-inflated recon does not dilute anti-collapse
export LAM_VAR_LAMBDA="1.0"       # keep the stabilized variance penalty
export LAM_MOTION_LAMBDA="5.0"   # gentler (was 10, collapsed); + stronger entropy below
echo "LARGE action with MOTION-WEIGHTED loss (lambda=10) + joint-entropy + var=1"
nvidia-smi -L
torchrun --standalone --nproc_per_node=4 scripts/train_latent_actions.py \
  --config configs/latent_actions_large.yaml --training_config configs/training_ddp.yaml
