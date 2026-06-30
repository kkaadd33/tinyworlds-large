#!/bin/bash
#SBATCH --job-name=evalgpu
#SBATCH --partition=ugrip
#SBATCH -q prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tinyworlds
cd ~/ugrip/tinyworlds-large
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/ugrip/tinyworlds-large:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VT=results/large_zelda/video_tokenizer/checkpoints/video_tokenizer_step_37500
LA=results/large_zelda/latent_actions/checkpoints/latent_actions_step_9500
DY=results/large_zelda/dynamics/checkpoints/dynamics_step_29500
echo "FINAL eval large dynamics @29500 (GPU): ZELDA in-dist + PONG/SONIC/PICODOOM OOD, with visualizations"
nvidia-smi -L
python scripts/evaluate.py --config configs/evaluation.yaml -- \
  use_latest_checkpoints=false \
  video_tokenizer_path=$VT latent_actions_path=$LA dynamics_path=$DY \
  eval_datasets=[ZELDA,PONG,SONIC,PICODOOM] n_clips_per_dataset=32 \
  n_visualization_clips=4 n_random_seeds=5 batch_size=8
