"""Diagnose whether the LAM collapses to a small set of action codes on OOD datasets.

This explains the negative Delta_4 PSNR observed on PONG: if the LAM encoder maps
all OOD transitions to (a few) saturated FSQ corners, the dynamics is driven by a
strong-but-wrong action signal, so PSNR can be *worse* than uniform random actions
which average across the whole codebook and tend toward a neutral transition.

Reports per dataset:
  * unique action codes used (out of n_actions)
  * top-5 action codes by frequency
  * entropy in nats and as fraction of log(n_actions)
"""

import os
import math
from collections import Counter
from typing import List

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.data_utils import load_data_and_data_loaders
from utils.config import EvaluationConfig, load_config
from utils.inference_utils import load_models
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


@torch.no_grad()
def _action_index_histogram(latent_action_model, loader: DataLoader, device: str) -> List[int]:
    """Run the LAM encoder over each clip and record the FSQ index of every inferred action."""
    counter: Counter = Counter()
    for x, _ in tqdm(loader, desc="lam encode", leave=False):
        x = x.to(device, non_blocking=True)
        actions_lat = latent_action_model.encode(x)  # [B, T-1, A] in {-1,+1}^A
        action_idx = latent_action_model.quantizer.get_indices_from_latents(actions_lat)  # [B, T-1]
        counter.update(action_idx.flatten().tolist())
    return counter


def main():
    args: EvaluationConfig = load_config(
        EvaluationConfig,
        default_config_path=os.path.join(os.getcwd(), 'configs', 'evaluation.yaml'),
    )
    args = _resolve(args)

    video_tokenizer, latent_action_model, dynamics_model = load_models(
        args.video_tokenizer_path, args.latent_actions_path, args.dynamics_path,
        args.device, use_actions=True,
    )
    n_actions = int(latent_action_model.quantizer.codebook_size)
    print(f"n_actions (codebook size) = {n_actions}")
    print(f"action_dim = log2(n_actions) = {int(math.log2(n_actions))}")

    print(f"\n{'Dataset':<14} {'unique':<8} {'usage%':<8} {'top1 idx':<10} {'top1 freq%':<11} {'entropy(nats)':<14} {'entropy/max':<11}")
    print("-" * 84)

    frames_per_clip = args.context_window + args.T_pred
    for dataset_name in args.eval_datasets:
        loader = _build_loader(dataset_name, args, frames_per_clip)
        counter = _action_index_histogram(latent_action_model, loader, args.device)
        total = sum(counter.values())
        unique = len(counter)

        items = counter.most_common()
        top1_idx, top1_freq = items[0]
        probs = [c / total for _, c in items]
        ent = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        max_ent = math.log(n_actions)

        print(
            f"{dataset_name:<14} {unique:<8d} {100.0 * unique / n_actions:<8.1f} "
            f"{top1_idx:<10d} {100.0 * top1_freq / total:<11.2f} "
            f"{ent:<14.3f} {ent / max_ent:<11.3f}"
        )


if __name__ == "__main__":
    main()
