#!/bin/bash
#SBATCH --job-name=evalcpu
#SBATCH --partition=ugrip
#SBATCH -q prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tinyworlds
cd ~/ugrip/tinyworlds-large
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/ugrip/tinyworlds-large:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""   # force CPU; no GPU requested so we don't hit the 4-GPU cap
DYN_STEP="${DYN_STEP:-10000}"
VT=results/large_zelda/video_tokenizer/checkpoints/video_tokenizer_step_37500
LA=results/large_zelda/latent_actions/checkpoints/latent_actions_step_9500
DY=results/large_zelda/dynamics/checkpoints/dynamics_step_${DYN_STEP}
echo "eval large dynamics @${DYN_STEP}  (CPU, ZELDA holdout)"
python scripts/evaluate.py --config configs/evaluation.yaml -- \
  use_latest_checkpoints=false device=cpu amp=false \
  video_tokenizer_path=$VT latent_actions_path=$LA dynamics_path=$DY \
  eval_datasets=[ZELDA] n_clips_per_dataset=8 n_random_seeds=2 batch_size=8 n_visualization_clips=4
