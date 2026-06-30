#!/bin/bash
#SBATCH --job-name=dynsharp
#SBATCH --partition=ugrip
#SBATCH -q prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tinyworlds
cd ~/ugrip/tinyworlds-large
export WANDB_MODE=online PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/ugrip/tinyworlds-large:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FILM_INIT_STD="0.1"   # FIX: dynamics action injection was std=1e-3 (~identity) -> action ignored. Raise it so the dynamics actually uses the action.
export DYN_ACTION_DROPOUT="0.15"   # CFG training: drop the action 15% of the time so the model learns p(x|a) AND p(x)
export NG_RUN_ROOT_DIR="results/large_zelda_ctx4_sharp_dyn"
export WANDB_RUN_GROUP="large_zelda_ctx4_fixaction_dyn"
echo "ctx4 dynamics (512/2048/18, lr0.0002) on the FIXED ctx4 action model (LAM-controllable +1.37) -- with FiLM 0.1 + action-dropout 0.15 (CFG training) -- on the SHARPER perceptual tokenizer (final pipeline: control + sharpness)"
nvidia-smi -L
torchrun --standalone --nproc_per_node=4 scripts/train_dynamics.py \
  --config configs/dynamics_large_ctx4_sharp.yaml --training_config configs/training_ddp.yaml
