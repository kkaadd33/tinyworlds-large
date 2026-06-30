#!/bin/bash
#SBATCH --job-name=toklargep
#SBATCH --partition=ugrip
#SBATCH -q shqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:8
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tinyworlds
cd ~/ugrip/tinyworlds-large
export WANDB_MODE=online PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/ugrip/tinyworlds-large:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NG_RUN_ROOT_DIR="results/large_zelda_tok_large_perceptual"
export WANDB_RUN_GROUP="large_zelda_tok_large_perceptual"
export TOK_PERCEPTUAL_WEIGHT="0.1"   # perceptual sharpness on top of the bigger model
echo "LARGE tokenizer (embed512/2048/8blk, ~Nx params) + VGG perceptual, 8xA100 via shqos -- max sharpness"
nvidia-smi -L
torchrun --standalone --nproc_per_node=8 scripts/train_video_tokenizer.py \
  --config configs/video_tokenizer_large.yaml --training_config configs/training_ddp_8gpu.yaml
