#!/bin/bash
#SBATCH --job-name=actlarge
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
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/ugrip/tinyworlds-large:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NG_RUN_ROOT_DIR="${RUN_DIR:-results/large_zelda}"
export WANDB_RUN_GROUP="${GROUP:-large_zelda}"
export LAM_ENTROPY_LAMBDA="${LAM_ENTROPY_LAMBDA:-0.1}"   # joint-entropy form (the fix); 0.1 was proven on tiny model
export LAM_VAR_LAMBDA="${LAM_VAR_LAMBDA:-1.0}"           # was 100 (destabilizing spikes); gentle floor now, entropy loss does anti-collapse
echo "RUN_DIR=$NG_RUN_ROOT_DIR  (LARGE action: embed512/hidden2048/8blocks, batch64x4=256)"
nvidia-smi -L
torchrun --standalone --nproc_per_node=4 scripts/train_latent_actions.py \
  --config configs/latent_actions_large.yaml --training_config configs/training_ddp.yaml
