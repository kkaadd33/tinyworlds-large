"""Decompose why autoregressive rollouts degrade.

For each held-out sequence we measure, in dB of PSNR vs ground truth, the loss at each
link of the rollout chain:

  CEILING        PSNR( detok(tok(x_t)), x_t )                 best a rollout could reach
  ONE-STEP TF    predict x_t from GROUND-TRUTH context        single-step dynamics quality
  PIXEL ROLLOUT  current method: decode each prediction to    @5/10/20 steps
                 pixels and re-tokenize as next context
  LATENT ROLLOUT keep latents, never round-trip through       @5/10/20 steps
                 pixels (isolates re-tokenization drift)

  RETOKEN FLIP   fraction of tokens that change under one      tokenizer idempotency
                 decode->encode round trip
"""
import argparse, os, torch
from datasets.data_utils import load_data_and_data_loaders
from utils.inference_utils import load_models
from utils.utils import find_latest_checkpoint


def to01(x): return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
def psnr(a, b):  # [B,C,H,W] in [0,1] -> [B]
    mse = ((to01(a) - to01(b)) ** 2).flatten(1).mean(1).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


@torch.no_grad()
def predict_step(dyn, tokenizer, ctx_lat, cond):
    out = dyn.forward_inference(context_latents=ctx_lat, prediction_horizon=1, num_steps=10,
                                index_to_latents_fn=lambda i: tokenizer.quantizer.get_latents_from_indices(i, dim=-1),
                                conditioning=cond, temperature=0.0)
    return out[:, -1:]  # [B,1,P,L] predicted latent for the next frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True); p.add_argument('--dataset', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--context', type=int, default=4); p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=6); p.add_argument('--batches', type=int, default=2)
    p.add_argument('--preload_ratio', type=float, default=0.15)
    p.add_argument('--dyn_ckpt', default=None, help='pin a specific dynamics checkpoint (matched-step comparison)')
    p.add_argument('--label', default=None)
    p.add_argument('--no_rollout', action='store_true', help='skip the 2 slow rollouts; only ceiling + single-step')
    args = p.parse_args()
    torch.set_grad_enabled(False)  # inference only; prevents graph buildup / OOM
    name = args.label or os.path.basename(args.run_dir.rstrip('/'))
    vt = find_latest_checkpoint('.', 'video_tokenizer', run_root_dir=args.run_dir, stage_name='video_tokenizer')
    la = find_latest_checkpoint('.', 'latent_actions', run_root_dir=args.run_dir, stage_name='latent_actions')
    dy = args.dyn_ckpt or find_latest_checkpoint('.', 'dynamics', run_root_dir=args.run_dir, stage_name='dynamics')
    print(f"dynamics ckpt: {dy}", flush=True)
    tok, lam, dyn = load_models(vt, la, dy, args.device, use_actions=True)
    lam.train()
    ctx, H, dev = args.context, args.horizon, args.device

    need = ctx + H + 1
    _, _, _, loader, _ = load_data_and_data_loaders(dataset=args.dataset, batch_size=args.batch_size,
                                                    num_frames=need, preload_ratio=args.preload_ratio)

    def cond_for(frames, i):  # action latents for predicting absolute frame ctx+i (teacher actions)
        return lam.encode(frames[:, i:i + ctx + 2])

    ceil = onestep = retoken = 0.0; n = 0
    pix = torch.zeros(H); lat = torch.zeros(H)
    it = iter(loader)
    for _ in range(args.batches):
        try: batch = next(it)
        except StopIteration: break
        frames = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(dev)
        if frames.shape[1] < need: continue
        B = frames.shape[0]; n += B
        gt_future = frames[:, ctx:ctx + H]

        # CEILING + RETOKEN FLIP (on the future frames)
        for t in range(H):
            x = frames[:, ctx + t:ctx + t + 1]
            idx1 = tok.tokenize(x)
            rec1 = tok.detokenize(tok.quantizer.get_latents_from_indices(idx1))
            ceil += psnr(rec1[:, 0], x[:, 0]).sum().cpu()
            idx2 = tok.tokenize(rec1)
            retoken += (idx1 != idx2).float().mean().item() * B
        # ONE-STEP TEACHER-FORCED: predict each future frame from GROUND-TRUTH context
        for t in range(H):
            gt_ctx = frames[:, t:t + ctx]
            ctx_lat = tok.quantizer.get_latents_from_indices(tok.tokenize(gt_ctx))
            pred_lat = predict_step(dyn, tok, ctx_lat, cond_for(frames, t))
            pred = tok.detokenize(pred_lat)[:, -1]
            onestep += psnr(pred, frames[:, ctx + t]).sum().cpu()

        # PIXEL ROLLOUT (current method) and LATENT ROLLOUT (no re-tokenization)
        if not args.no_rollout:
            gen = frames[:, :ctx].clone()
            lat_buf = tok.quantizer.get_latents_from_indices(tok.tokenize(frames[:, :ctx]))
            for i in range(H):
                c = cond_for(frames, i)
                # pixel: re-tokenize generated frames each step
                pl = predict_step(dyn, tok, tok.quantizer.get_latents_from_indices(tok.tokenize(gen[:, -ctx:])), c)
                gen = torch.cat([gen, tok.detokenize(pl)[:, -1:]], dim=1)
                pix[i] += psnr(gen[:, -1], gt_future[:, i]).sum().cpu()
                # latent: keep latents, never decode/re-encode within the loop
                ll = predict_step(dyn, tok, lat_buf[:, -ctx:], c)
                lat_buf = torch.cat([lat_buf, ll], dim=1)
                lat_dec = tok.detokenize(lat_buf[:, -1:])[:, -1]
                lat[i] += psnr(lat_dec, gt_future[:, i]).sum().cpu()

    if n == 0: raise RuntimeError("no usable batches")
    ceil /= n * H; onestep /= n * H; retoken /= n * H
    pix /= n; lat /= n

    print(f"\n==== Rollout diagnostics: {name} ({args.dataset}), {n} sequences ====")
    print(f"tokenizer CEILING  (recon PSNR)        : {ceil:.2f} dB")
    print(f"ONE-STEP teacher-forced PSNR           : {onestep:.2f} dB   (gap to ceiling: {ceil-onestep:.2f} dB)")
    if not args.no_rollout:
        print(f"PIXEL  rollout  PSNR @5/10/20          : {pix[4]:.2f} / {pix[9]:.2f} / {pix[19]:.2f} dB")
        print(f"LATENT rollout  PSNR @5/10/20          : {lat[4]:.2f} / {lat[9]:.2f} / {lat[19]:.2f} dB")
        print(f"   single-step loss {ceil-onestep:.1f} | accumulation(latent) {onestep-lat[19]:.1f} | re-tokenization {lat[19]-pix[19]:.1f}")


if __name__ == "__main__":
    main()
