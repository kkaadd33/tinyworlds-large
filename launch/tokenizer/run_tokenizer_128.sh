#!/bin/bash
#SBATCH --job-name=tok128
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
export NG_RUN_ROOT_DIR="results/large_zelda_128_tok"
export WANDB_RUN_GROUP="large_zelda_128_tok"
export TOK_PERCEPTUAL_WEIGHT="0.1"
echo "128-RES large tokenizer + perceptual, 4x80GB (higher resolution pipeline, stage 1/3)"
nvidia-smi -L
torchrun --standalone --nproc_per_node=4 scripts/train_video_tokenizer.py \
  --config configs/video_tokenizer_large_128.yaml --training_config configs/training_ddp_128.yaml
