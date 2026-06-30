#!/bin/bash
#SBATCH --job-name=tokpercept
#SBATCH --partition=ugrip
#SBATCH -q prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --time=16:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tinyworlds
cd ~/ugrip/tinyworlds-large
export WANDB_MODE=online PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/ugrip/tinyworlds-large:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NG_RUN_ROOT_DIR="results/large_zelda_tok_perceptual"
export WANDB_RUN_GROUP="large_zelda_tok_perceptual"
# PARAM-NEUTRAL sharpness fix: SAME tiny tokenizer (embed32/hid128/4blk) + train-only VGG perceptual loss.
# Pixel smooth_l1 alone -> blurry recons (31.45 dB ceiling); perceptual restores high-frequency detail. 0 inference params.
export TOK_PERCEPTUAL_WEIGHT="0.1"   # KEY hyperparameter -- tune vs the smooth_l1 magnitude (monitor recon doesn't destabilize)
echo "TINY tokenizer RETRAIN + VGG perceptual loss (w=$TOK_PERCEPTUAL_WEIGHT) -- param-neutral sharpness experiment"
nvidia-smi -L
torchrun --standalone --nproc_per_node=4 scripts/train_video_tokenizer.py \
  --config configs/video_tokenizer.yaml --training_config configs/training_ddp.yaml
