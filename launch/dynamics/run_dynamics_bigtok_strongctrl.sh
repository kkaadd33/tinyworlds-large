#!/bin/bash
#SBATCH --job-name=dynstrong
#SBATCH --partition=ugrip_80
#SBATCH -q prod
#SBATCH --nodelist=gpu07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tinyworlds
cd ~/ugrip/tinyworlds-large
export WANDB_MODE=online PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/ugrip/tinyworlds-large:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NG_RUN_ROOT_DIR="results/large_zelda_bigtok_strongctrl_dyn"
export WANDB_RUN_GROUP="large_zelda_bigtok_strongctrl_dyn"
export FILM_INIT_STD="0.2"          # stronger action injection (was 0.1)
export DYN_ACTION_DROPOUT="0.25"    # more aggressive CFG training (was 0.15) -> stronger action reliance
echo "STRONG-CONTROL dynamics on sharp tokenizer (FiLM 0.2 + dropout 0.25) -- max controllability"
nvidia-smi -L
torchrun --standalone --nproc_per_node=4 scripts/train_dynamics.py \
  --config configs/dynamics_large_ctx4_bigtok.yaml --training_config configs/training_ddp.yaml
