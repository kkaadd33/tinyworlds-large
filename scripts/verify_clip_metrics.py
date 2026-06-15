"""Verify the eval metrics on a single saved-visualization clip.

Reproduces evaluate.py's exact rollout for one (dataset, clip_idx) pair, then
computes per-frame metrics three ways:
  * our `frame_psnr` (utils.metrics)
  * skimage.metrics.peak_signal_noise_ratio   <-- independent reference
  * skimage.metrics.structural_similarity     <-- perceptual sanity check (PSNR is MSE-based)

Run:
    python scripts/verify_clip_metrics.py --config configs/evaluation.yaml \
        -- dataset=SONIC clip_idx=3
"""

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from torch.utils.data import DataLoader, Subset

from datasets.data_utils import load_data_and_data_loaders
from utils.config import EvaluationConfig, load_config
from utils.evaluation_utils import rollout
from utils.inference_utils import load_models
from utils.metrics import frame_psnr
from utils.utils import find_latest_checkpoint


def _missing(p):
    return p is None or not os.path.exists(p)


def _resolve(args: EvaluationConfig):
    base = os.getcwd()
    if args.use_latest_checkpoints or _missing(args.video_tokenizer_path):
        args.video_tokenizer_path = find_latest_checkpoint(base, "video_tokenizer")
    if args.use_latest_checkpoints or _missing(args.latent_actions_path):
        args.latent_actions_path = find_latest_checkpoint(base, "latent_actions")
    if args.use_latest_checkpoints or _missing(args.dynamics_path):
        args.dynamics_path = find_latest_checkpoint(base, "dynamics")
    return args


def _build_loader(dataset_name: str, args: EvaluationConfig, frames_per_clip: int) -> DataLoader:
    is_zelda = dataset_name == 'ZELDA'
    use_holdout = is_zelda and args.zelda_use_holdout
    train_set, val_set, _, _, _ = load_data_and_data_loaders(
        dataset=dataset_name,
        batch_size=args.batch_size,
        num_frames=frames_per_clip,
        disable_test_split=not use_holdout,
        resize_to=tuple(args.force_resize_to),
    )
    ds = val_set if use_holdout else train_set
    n = args.n_clips_per_dataset
    if n is not None and n < len(ds):
        gen = torch.Generator(device='cpu').manual_seed(int(args.seed))
        perm = torch.randperm(len(ds), generator=gen)[:n].tolist()
        ds = Subset(ds, perm)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)


def _denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1] clamped, on CPU as float32."""
    return ((x.detach().to(torch.float32).cpu() + 1.0) / 2.0).clamp(0.0, 1.0)


def _to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    """[C,H,W] in [0,1] -> uint8 HWC for skimage."""
    arr = (frame_chw.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return arr


def main():
    # Custom CLI: pull `dataset` and `clip_idx` out before evaluate.py-style override parsing
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=os.path.join(os.getcwd(), 'configs', 'evaluation.yaml'))
    pre.add_argument('overrides', nargs=argparse.REMAINDER)
    pre_args = pre.parse_args()

    dataset = "SONIC"
    clip_idx = 3
    other_overrides = []
    for tok in (pre_args.overrides or []):
        if tok.startswith("dataset="):
            dataset = tok.split("=", 1)[1]
        elif tok.startswith("clip_idx="):
            clip_idx = int(tok.split("=", 1)[1])
        else:
            other_overrides.append(tok)

    # rebuild argv so load_config sees just the rollout/eval overrides
    sys.argv = ["verify_clip_metrics.py", "--config", pre_args.config] + other_overrides
    args: EvaluationConfig = load_config(EvaluationConfig)
    args = _resolve(args)

    print(f"=== verify_clip_metrics: {dataset} clip {clip_idx} ===")
    print(f"checkpoints: VT={os.path.basename(args.video_tokenizer_path)}, "
          f"LAM={os.path.basename(args.latent_actions_path)}, "
          f"Dyn={os.path.basename(args.dynamics_path)}")

    video_tokenizer, latent_action_model, dynamics_model = load_models(
        args.video_tokenizer_path, args.latent_actions_path, args.dynamics_path,
        args.device, use_actions=True,
    )
    n_actions = int(latent_action_model.quantizer.codebook_size)

    frames_per_clip = args.context_window + args.T_pred
    loader = _build_loader(dataset, args, frames_per_clip)

    # Walk to the batch that contains the requested clip and pull just that one out
    target_batch = clip_idx // args.batch_size
    in_batch_idx = clip_idx % args.batch_size
    print(f"resolving clip {clip_idx}: batch_idx={target_batch}, in_batch_idx={in_batch_idx}, batch_size={args.batch_size}")

    x_batch = None
    for b_idx, (x, _) in enumerate(loader):
        if b_idx == target_batch:
            x_batch = x.to(args.device)
            break
    if x_batch is None:
        raise RuntimeError(f"clip {clip_idx} not found in loader (loader has < {target_batch + 1} batches)")

    # Run both rollouts (use s=0 random seed to match what was visualized in PING)
    rnd_seed_s0 = int(args.seed) + 7919 * 1 + target_batch
    print(f"random seed (s=0): {rnd_seed_s0}")

    x_hat = rollout(
        video_tokenizer, latent_action_model, dynamics_model, x_batch,
        action_mode='gt_lam',
        n_actions=n_actions,
        context_window=args.context_window, T_pred=args.T_pred,
        prediction_horizon=args.prediction_horizon,
        num_steps=args.num_maskgit_steps,
        temperature=args.temperature,
    )
    x_hat_prime = rollout(
        video_tokenizer, latent_action_model, dynamics_model, x_batch,
        action_mode='random',
        n_actions=n_actions,
        context_window=args.context_window, T_pred=args.T_pred,
        prediction_horizon=args.prediction_horizon,
        num_steps=args.num_maskgit_steps,
        temperature=args.temperature,
        action_seed=rnd_seed_s0,
    )

    # ---- Slice to the requested clip ----
    x_clip = x_batch[in_batch_idx : in_batch_idx + 1]                 # [1, T_total, C, H, W]
    target = x_clip[:, args.context_window:]                           # [1, T_pred,  C, H, W]
    x_hat_clip = x_hat[in_batch_idx : in_batch_idx + 1]                # [1, T_pred,  C, H, W]
    x_hat_prime_clip = x_hat_prime[in_batch_idx : in_batch_idx + 1]    # [1, T_pred,  C, H, W]

    # ---- our PSNR (utils.metrics.frame_psnr) ----
    our_psnr_gt = frame_psnr(x_hat_clip, target)[0].cpu().tolist()       # [T_pred]
    our_psnr_rnd = frame_psnr(x_hat_prime_clip, target)[0].cpu().tolist()

    # ---- skimage PSNR + SSIM (per frame) on uint8 [0, 255] ----
    target_d = _denorm(target)[0]            # [T_pred, C, H, W] in [0,1]
    x_hat_d = _denorm(x_hat_clip)[0]
    x_hat_prime_d = _denorm(x_hat_prime_clip)[0]

    sk_psnr_gt, sk_psnr_rnd = [], []
    sk_ssim_gt, sk_ssim_rnd = [], []
    for t in range(args.T_pred):
        gt_u8 = _to_uint8_hwc(target_d[t])
        gt_lam_u8 = _to_uint8_hwc(x_hat_d[t])
        rnd_u8 = _to_uint8_hwc(x_hat_prime_d[t])

        sk_psnr_gt.append(sk_psnr(gt_u8, gt_lam_u8, data_range=255))
        sk_psnr_rnd.append(sk_psnr(gt_u8, rnd_u8, data_range=255))
        sk_ssim_gt.append(sk_ssim(gt_u8, gt_lam_u8, channel_axis=-1, data_range=255))
        sk_ssim_rnd.append(sk_ssim(gt_u8, rnd_u8, channel_axis=-1, data_range=255))

    # ---- Pretty print ----
    header = f"{'t':<3} {'ours_psnr_gt':>13} {'sk_psnr_gt':>11} {'ours_psnr_rnd':>14} {'sk_psnr_rnd':>12} {'ssim_gt':>9} {'ssim_rnd':>10}"
    print()
    print(header)
    print("-" * len(header))
    for t in range(args.T_pred):
        print(
            f"{t:<3d} "
            f"{our_psnr_gt[t]:>13.3f} {sk_psnr_gt[t]:>11.3f} "
            f"{our_psnr_rnd[t]:>14.3f} {sk_psnr_rnd[t]:>12.3f} "
            f"{sk_ssim_gt[t]:>9.4f} {sk_ssim_rnd[t]:>10.4f}"
        )

    def _mean(xs):
        return sum(xs) / len(xs)

    print("-" * len(header))
    print(
        f"{'mean':<3} "
        f"{_mean(our_psnr_gt):>13.3f} {_mean(sk_psnr_gt):>11.3f} "
        f"{_mean(our_psnr_rnd):>14.3f} {_mean(sk_psnr_rnd):>12.3f} "
        f"{_mean(sk_ssim_gt):>9.4f} {_mean(sk_ssim_rnd):>10.4f}"
    )

    if args.T_pred >= 4:
        print()
        print(f"Delta_4 PSNR (ours): {our_psnr_gt[3] - our_psnr_rnd[3]:+.3f}")
        print(f"Delta_4 PSNR (sk):   {sk_psnr_gt[3] - sk_psnr_rnd[3]:+.3f}")
        print(f"Delta_4 SSIM:        {sk_ssim_gt[3] - sk_ssim_rnd[3]:+.4f}")


if __name__ == "__main__":
    main()
