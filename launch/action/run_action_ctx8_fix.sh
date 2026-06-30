#!/bin/bash
#SBATCH --job-name=actfix8
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
export NG_RUN_ROOT_DIR="results/large_zelda_ctx8_fix2"
export WANDB_RUN_GROUP="large_zelda_ctx8_fix2"
export LAM_KEEP_RATE="1.0"        # THE FIX: decoder sees the real previous frame (was 0.0)
export LAM_ENTROPY_LAMBDA="0.1"
export LAM_VAR_LAMBDA="1.0"
export LAM_MOTION_LAMBDA="0.0"    # motion off -> isolate the keep_rate fix
export LAM_PREDICT_DELTA="1"        # FIX #2: predict residual/delta, closes the copy-frame shortcut
echo "ACTION ctx8 WITH KEEP_RATE FIX (decoder sees previous frame; action on critical path)"
nvidia-smi -L
torchrun --standalone --nproc_per_node=4 scripts/train_latent_actions.py \
  --config configs/latent_actions_large_ctx8.yaml --training_config configs/training_ddp_ctx8.yaml
