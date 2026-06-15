"""Per-frame PSNR utilities for the Genie-style evaluation pipeline.

Frames inside the model live in [-1, 1] because of `transforms.Normalize((0.5,)*3, (0.5,)*3)`
in `datasets/data_utils.py`. PSNR is conventionally reported on [0, 1]-normalized images, so
every helper denormalizes first and clamps to the valid range before computing MSE.
"""

import torch


def denormalize(x: torch.Tensor) -> torch.Tensor:
    # frames live in [-1, 1] inside the model; PSNR convention is [0, 1]
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def frame_psnr(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    # pred, gt: [B, T, C, H, W] in [-1, 1] (model output convention)
    # returns:  [B, T] PSNR per (batch, frame) in dB on [0, 1]-rescaled inputs
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs gt {tuple(gt.shape)}")
    if pred.dim() != 5:
        raise ValueError(f"expected [B, T, C, H, W]; got {tuple(pred.shape)}")

    pred = denormalize(pred.detach().to(torch.float32))
    gt = denormalize(gt.detach().to(torch.float32))

    mse = ((pred - gt) ** 2).mean(dim=(-1, -2, -3))  # [B, T]
    return 10.0 * torch.log10(1.0 / mse.clamp_min(eps))


def aggregate_psnr_per_t(psnr_bt: torch.Tensor) -> torch.Tensor:
    # mean over batch dim only; preserves per-t resolution
    if psnr_bt.dim() != 2:
        raise ValueError(f"expected [B, T]; got {tuple(psnr_bt.shape)}")
    return psnr_bt.mean(dim=0)
