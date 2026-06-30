"""LAM-only controllability test (isolates the action model, no dynamics needed).

Encode real clips -> latent actions -> decode with (a) the TRUE actions vs
(b) SHUFFLED actions (each clip decoded with another clip's action sequence) vs
(c) RANDOM action latents. If TRUE >> SHUFFLED/RANDOM in reconstruction PSNR,
the action codes genuinely steer the output => the model is controllable.
"""
import os, torch
from models.latent_actions import LatentActionModel
from datasets.data_utils import load_data_and_data_loaders
from utils.utils import load_latent_actions_from_checkpoint
from utils.metrics import frame_psnr

device = 'cpu'
ckpt = os.environ['CKPT']
n_batches = int(os.environ.get('N_BATCHES', '6'))
bs = int(os.environ.get('BS', '16'))

# architecture must match latent_actions_large.yaml (frame 64, patch 4, embed 512, 8 heads, hidden 2048, 8 blocks, 4 actions)
model = LatentActionModel(frame_size=(64, 64), patch_size=4, embed_dim=512,
                          num_heads=8, hidden_dim=2048, num_blocks=8, n_actions=4).to(device)
model, _ = load_latent_actions_from_checkpoint(ckpt, device, model, False)
model.eval()
print(f"loaded: {ckpt}")
print(f"LAM_PREDICT_DELTA={os.environ.get('LAM_PREDICT_DELTA','0')}  (must match training)")

nframes = int(os.environ.get('NFRAMES', '4'))
_, _, loader, _, _ = load_data_and_data_loaders(dataset='ZELDA', batch_size=bs, num_frames=nframes)

tot_true = tot_shuf = tot_rand = 0.0
n = 0
codes_seen = set()
with torch.no_grad():
    it = iter(loader)
    for b in range(n_batches):
        x, _ = next(it)
        x = x.to(device)                                  # [B,4,C,H,W]
        B = x.shape[0]
        a_true = model.encoder(x)                         # [B,3,A]
        q_true = model.quantizer(a_true)                  # [B,3,A]
        # log which discrete codes actually fire
        idx = ((torch.tanh(a_true) > 0).long() * (2 ** torch.arange(a_true.shape[-1]))).sum(-1)
        codes_seen.update(idx.unique().tolist())
        # nulls
        perm = torch.randperm(B)
        q_shuf = q_true[perm]                             # another clip's actions
        a_rand = torch.randn_like(a_true) * (1.3 ** 0.5)  # random latents at the healthy variance
        q_rand = model.quantizer(a_rand)
        target = x[:, 1:]                                 # [B,3,C,H,W]
        r_true = model.decoder(x, q_true, training=False)
        r_shuf = model.decoder(x, q_shuf, training=False)
        r_rand = model.decoder(x, q_rand, training=False)
        tot_true += frame_psnr(r_true, target).mean().item() * B
        tot_shuf += frame_psnr(r_shuf, target).mean().item() * B
        tot_rand += frame_psnr(r_rand, target).mean().item() * B
        n += B

print(f"\nclips evaluated: {n}")
print(f"distinct action codes used: {sorted(codes_seen)} ({len(codes_seen)}/4)")
print(f"PSNR  true-actions   : {tot_true/n:6.3f} dB")
print(f"PSNR  shuffled-actions: {tot_shuf/n:6.3f} dB")
print(f"PSNR  random-actions : {tot_rand/n:6.3f} dB")
print(f"Delta (true - shuffled): {(tot_true-tot_shuf)/n:+.3f} dB")
print(f"Delta (true - random)  : {(tot_true-tot_rand)/n:+.3f} dB")
print("\n=> controllable if the Deltas are clearly positive (true actions reconstruct better).")
