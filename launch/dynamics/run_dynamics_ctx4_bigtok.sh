#!/bin/bash
#SBATCH --job-name=dynbigtok
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
export NG_RUN_ROOT_DIR="results/large_zelda_ctx4_bigtok_dyn"
export WANDB_RUN_GROUP="large_zelda_ctx4_bigtok_dyn"
export FILM_INIT_STD="0.1"          # controllability: strong action injection
export DYN_ACTION_DROPOUT="0.15"    # controllability: CFG training
echo "ctx4 dynamics on the LARGE (sharp) tokenizer + control recipe -- final pipeline (control + sharpness)"
nvidia-smi -L
torchrun --standalone --nproc_per_node=4 scripts/train_dynamics.py \
  --config configs/dynamics_large_ctx4_bigtok.yaml --training_config configs/training_ddp.yaml
