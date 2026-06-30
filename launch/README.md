# Launch scripts

Slurm / torchrun launch scripts, grouped by training stage:

- `tokenizer/` — video tokenizer runs (incl. the perceptual-loss and large variants)
- `action/` — latent action model runs (incl. the collapse-fix variance-band / low-lr variants)
- `dynamics/` — dynamics runs (incl. CFG / FiLM / strong-control / context-corruption variants)
- `eval/` — evaluation jobs (CPU / GPU)

Each script `cd`s to the repo root and calls the matching `scripts/train_*.py` with a config from `configs/`,
so they can be moved/grouped freely. Submit with `sbatch launch/<stage>/<script>.sh`.

Key scripts for the chosen runs:
- Run B dynamics: `dynamics/run_dynamics_ctx4_sharp.sh`
- Run D dynamics: `dynamics/run_dynamics_bigtok_strongctrl.sh`
- Perceptual tokenizer: `tokenizer/run_tokenizer_perceptual.sh` (tiny) / `tokenizer/run_tokenizer_large_perceptual.sh` (big)
