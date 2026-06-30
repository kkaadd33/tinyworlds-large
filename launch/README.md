# Launch scripts

Slurm / torchrun launch scripts, grouped by training stage:

- `tokenizer/` — video tokenizer runs
- `action/` — latent action model (LAM) runs
- `dynamics/` — dynamics runs
- `eval/` — evaluation jobs (CPU / GPU)

Each script `cd`s to the repo root and calls the matching `scripts/train_*.py` with a config from
`configs/`, so they can be moved/grouped freely. Submit with `sbatch launch/<stage>/<script>.sh`.

## Which scripts actually produced the runs (A / B / C / D)

These are the **only** scripts behind the runs shown in
[`../report_figures/runs/`](../report_figures/runs). All four share the same fixed action model (LAM).
Mapping verified from each script's `NG_RUN_ROOT_DIR` output directory.

| Run | Tokenizer | Action (LAM) | Dynamics |
|-----|-----------|--------------|----------|
| A — tiny, no perceptual | `tokenizer/run_tokenizer.sh` | `action/run_action_ctx4_lr.sh` | `dynamics/run_dynamics_ctx4_cfg.sh` |
| **B — tiny + perceptual** | `tokenizer/run_tokenizer_perceptual.sh` | `action/run_action_ctx4_lr.sh` | `dynamics/run_dynamics_ctx4_sharp.sh` |
| C — big tokenizer | `tokenizer/run_tokenizer_large_perceptual.sh` | `action/run_action_ctx4_lr.sh` | `dynamics/run_dynamics_ctx4_bigtok.sh` |
| **D — big + strong control** | `tokenizer/run_tokenizer_large_perceptual.sh` | `action/run_action_ctx4_lr.sh` | `dynamics/run_dynamics_bigtok_strongctrl.sh` |

`run_action_ctx4_lr.sh` is the LAM-collapse fix (low LR + variance band) and is **shared by all four runs**.
Evaluation of these runs was done with `eval/run_myeval_gpu.sh` (or `scripts/evaluate.py` directly).

## Everything else is an abandoned experiment

Every other script under `launch/` is an earlier attempt that was superseded — different context length,
128 resolution, 16 actions, intermediate collapse-fix tries (`*_fix`, `*_varband`, `*_varceil`,
`*_research`), `ctx8` variants, etc. They are kept only for history and are **not** part of the
A/B/C/D results.
