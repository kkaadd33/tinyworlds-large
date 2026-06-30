#!/bin/bash
#SBATCH --job-name=actctx8fix
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
export NG_RUN_ROOT_DIR="results/large_zelda_ctx8_fix"
export WANDB_RUN_GROUP="large_zelda_ctx8_fix"
# --- CORRECTED RECIPE (research-backed) ---
# Root cause: small-motion regime -> "copy the previous frame" is near-optimal under pixel loss,
# so the action gets almost no gradient and the 4-code FSQ codebook collapses. Fixes, combined:
export LAM_KEEP_RATE="0.5"        # PARTIAL Genie masking (sweet spot). 0.0=mask-all and 1.0=mask-none both collapse.
export FILM_INIT_STD="0.05"       # NEW: strong action injection. Original std=1e-3 ~ identity -> action ignorable at init.
export LAM_PREDICT_DELTA="1"      # residual: static background is free, action only needs to carry the change.
export LAM_ENTROPY_LAMBDA="0.1"   # MAGVIT-v2 joint code-entropy (keeps all 4 codes alive).
export LAM_VAR_LAMBDA="10.0"      # STRONG floor now (was a token 1.0)
export LAM_MOTION_LAMBDA="0.0"    # motion-weighting OFF for this clean run; add next if still static.
export LAM_VAR_MAX="1.5"          # NEW: cap pre-quant variance (anti-runaway). 1.47-2.53 was the stable range.
export LAM_VAR_TARGET="1.0"        # NEW: raise the FLOOR to 1.0. Band [1.0,1.5] brackets the healthy 1.2 -> variance cannot drift down to collapse.
echo "ACTION ctx4 CTX8 FIX: winning recipe (keep0.5 + FiLM0.05 + delta + var-band + lr3e-5) at context 8"
nvidia-smi -L
torchrun --standalone --nproc_per_node=4 scripts/train_latent_actions.py \
  --config configs/latent_actions_large_ctx8_lr3e5.yaml --training_config configs/training_ddp_ctx8.yaml
