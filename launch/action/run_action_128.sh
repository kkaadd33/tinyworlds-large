#!/bin/bash
#SBATCH --job-name=act128
#SBATCH --partition=ugrip_80
#SBATCH -q prod
#SBATCH --nodelist=gpu07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
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
export NG_RUN_ROOT_DIR="results/large_zelda_128_act"
export WANDB_RUN_GROUP="large_zelda_128_act"
export LAM_KEEP_RATE="0.5" FILM_INIT_STD="0.05" LAM_PREDICT_DELTA="1"
export LAM_ENTROPY_LAMBDA="0.1" LAM_VAR_LAMBDA="10.0" LAM_VAR_MAX="1.5" LAM_VAR_TARGET="1.0"
echo "128-RES action model, winning control recipe (stage 2/3)"
nvidia-smi -L
torchrun --standalone --nproc_per_node=4 scripts/train_latent_actions.py \
  --config configs/latent_actions_large_128.yaml --training_config configs/training_ddp_128.yaml
