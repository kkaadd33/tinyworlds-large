# Evaluation summary

- Checkpoints: VT=results/large_zelda_tok_perceptual/video_tokenizer/checkpoints/video_tokenizer_step_35000, LAM=results/large_zelda_ctx4_lr3e5/latent_actions/checkpoints/latent_actions_step_8000, Dyn=results/large_zelda_ctx4_sharp_dyn/dynamics/checkpoints/dynamics_step_29500
- context_window=2, T_pred=10, num_maskgit_steps=10, temperature=0.0, n_random_seeds=2

| Dataset | N clips | Recon PSNR (mean) | PSNR(x,x_hat) t=4 | PSNR(x,x_hat') t=4 | Delta_4 PSNR |
|---|---|---|---|---|---|
| ZELDA | 32 | 24.062 | 27.101 | 26.677 | +0.425 |
