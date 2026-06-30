# Evaluation summary

- Checkpoints: VT=results/large_zelda_tok_large_perceptual/video_tokenizer/checkpoints/video_tokenizer_step_37500, LAM=results/large_zelda_ctx4_lr3e5/latent_actions/checkpoints/latent_actions_step_8000, Dyn=results/large_zelda_ctx4_bigtok_dyn/dynamics/checkpoints/dynamics_step_29500
- context_window=2, T_pred=10, num_maskgit_steps=10, temperature=0.0, n_random_seeds=2

| Dataset | N clips | Recon PSNR (mean) | PSNR(x,x_hat) t=4 | PSNR(x,x_hat') t=4 | Delta_4 PSNR |
|---|---|---|---|---|---|
| ZELDA | 32 | 34.643 | 28.241 | 29.126 | -0.885 |
