#!/bin/bash
#SBATCH --job-name=toklarge
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
export NG_RUN_ROOT_DIR="results/large_zelda_tok"
export WANDB_RUN_GROUP="large_zelda_tok"
echo "LARGE tokenizer (512/2048/8, ctx8, batch12x4x2acc=96, lr0.0003, FSQ 1024)"
nvidia-smi -L
torchrun --standalone --nproc_per_node=4 scripts/train_video_tokenizer.py \
  --config configs/video_tokenizer_large.yaml --training_config configs/training_ddp_ctx8.yaml
