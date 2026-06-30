#!/bin/bash
#SBATCH --job-name=myeval
#SBATCH --partition=ugrip
#SBATCH -q prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tinyworlds
cd ~/ugrip/tinyworlds-large
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/ugrip/tinyworlds-large:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DY=results/large_zelda/dynamics/checkpoints/dynamics_step_29500
echo "MY eval (single-step + rollout decomposition) on large dynamics @29500, ZELDA, context=4 (training context)"
nvidia-smi -L
# tokenizer + action auto-resolved from results/large_zelda (latest = our trained ones)
python scripts/eval_rollout_diagnostics.py --run_dir results/large_zelda --dataset ZELDA \
  --context 4 --horizon 20 --dyn_ckpt $DY --label large@29500 --batch_size 8 --batches 12
