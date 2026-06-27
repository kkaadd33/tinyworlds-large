from models.utils import ModelType
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from einops import rearrange, repeat, reduce
from models.st_transformer import STTransformer, PatchEmbedding
from models.fsq import FiniteScalarQuantizer

NUM_LATENT_ACTIONS_BINS = 2

class LatentActionsEncoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8, 
                 hidden_dim=256, num_blocks=4, action_dim=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)
        
        # embeddings to discrete latent bottleneck actions
        self.action_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 4 * action_dim),
            nn.GELU(),
            nn.Linear(4 * action_dim, action_dim)
        )

    def forward(self, frames):
        # frames: [B, T, C, H, W]
        batch_size, seq_len, C, H, W = frames.shape

        embeddings = self.patch_embed(frames)  # [B, T, P, E]
        transformed = self.transformer(embeddings)

        # TODO: try attention pooling + mean instead of mean + concat
        # mean pool over patches (since one action per frame)
        pooled = transformed.mean(dim=2)  # [B, T, E]

        # combine features from current and next frame
        actions = []
        for t in range(seq_len - 1):
            # concat current and next frame features
            combined = torch.cat([pooled[:, t], pooled[:, t+1]], dim=1)  # [B, E*2]
            action = self.action_head(combined)  # [B, A]
            actions.append(action)

        actions = torch.stack(actions, dim=1)  # [B, T-1, A]

        return actions

class LatentActionsDecoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, num_heads=8,
                 hidden_dim=256, num_blocks=4, conditioning_dim=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True, conditioning_dim=conditioning_dim)

        # embeddings to mixed frame output patches
        self.frame_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3 * patch_size * patch_size),
            nn.Tanh()
        )

        self.frame_size = frame_size
        self.patch_size = patch_size
        self.num_patches = (frame_size[0] // patch_size) * (frame_size[1] // patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

    def forward(self, frames, actions, training=True):
        # frames: [B, T, C, H, W]
        # actions: [B, T - 1, A]
        B, T, C, H, W = frames.shape
        frames = frames[:, :-1] # [B, T-1, C, H, W]
        video_embeddings = self.patch_embed(frames)  # [B, T-1, P, E]
        _, _, P, E = video_embeddings.shape

        # mask certain tokens from all frames except first frame
        # this strongly forces actions to contain most useful info (I recommend to keep based on experiments)
        if training and self.training:
            # keep_rate = Genie-style masking bottleneck = fraction of previous-frame tokens KEPT (not masked).
            # 0.0 is NOT a bug: it is the original upstream default (AlmondGod/tinyworlds) -- mask every
            # non-anchor token so the action must carry the change. But BOTH extremes collapse, for opposite
            # reasons: 0.0 -> decoder sees only frame 0 and can't route it through 4 codes; 1.0 -> decoder
            # sees the full previous frame and just COPIES it, so the action is useless. The Genie sweet spot
            # is PARTIAL masking (~0.5). Default restored to upstream 0.0; sweep via LAM_KEEP_RATE.
            keep_rate = float(os.environ.get('LAM_KEEP_RATE', 0.0))
            keep = (torch.rand(B, T-1, P, 1, device=frames.device) < keep_rate)
            keep[:, 0] = 1  # never mask first frame tokens (anchor) TODO: try rid of ablation
            video_embeddings = torch.where(
                keep, video_embeddings,
                self.mask_token.to(video_embeddings.dtype).expand_as(video_embeddings)
            )

        transformed = self.transformer(video_embeddings, conditioning=actions)  # [B, T-1, P, E]
        patches = self.frame_head(transformed)  # [B, T-1, P, 3 * S * S]
        patches = rearrange(
            patches, 'b t p (c p1 p2) -> b t c p p1 p2', c=3, p1=self.patch_size, p2=self.patch_size
        ) # [B, T-1, C, P, S, S]
        pred_frames = rearrange(
            patches, 'b t c (h w) p1 p2 -> b t c (h p1) (w p2)', h=H//self.patch_size, w=W//self.patch_size
        ) # [B, T-1, C, H, W]
        # DELTA / residual prediction: treat the Tanh head output as the CHANGE from the previous frame,
        # not the absolute frame. The static background becomes free (delta=0 there), so the loss and the
        # action gradient concentrate on the moving region the action must explain -> closes the
        # "copy the previous frame" shortcut that lets the model ignore the action. env-gated.
        if os.environ.get('LAM_PREDICT_DELTA', '0') == '1':
            pred_frames = frames + pred_frames  # `frames` = the previous frames (sliced to [:, :-1] above)
        return pred_frames  # [B, T-1, C, H, W]

class LatentActionModel(nn.Module):
    def __init__(self, frame_size=(128, 128), n_actions=8, patch_size=8, embed_dim=128, 
                 num_heads=8, hidden_dim=256, num_blocks=4):
        super().__init__()
        assert math.log(n_actions, NUM_LATENT_ACTIONS_BINS).is_integer(), f"n_actions must be a power of {NUM_LATENT_ACTIONS_BINS}"
        self.action_dim=int(math.log(n_actions, NUM_LATENT_ACTIONS_BINS))
        self.encoder = LatentActionsEncoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, action_dim=self.action_dim)
        self.quantizer = FiniteScalarQuantizer(latent_dim=self.action_dim, num_bins=NUM_LATENT_ACTIONS_BINS)
        self.decoder = LatentActionsDecoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, conditioning_dim=self.action_dim)
        self.var_target = float(os.environ.get('LAM_VAR_TARGET', 0.01))
        # var_lambda=100 destabilizes: when encoder variance momentarily collapses, relu(0.01-var)*100
        # spikes the loss to ~0.5 and oscillates usage between 1 and 4 codes. The joint entropy loss
        # now does the real anti-collapse work, so keep only a gentle variance floor (env-overridable).
        self.var_lambda = float(os.environ.get('LAM_VAR_LAMBDA', 100.0))
        # variance CEILING (anti-runaway). Observed: pre-quant variance grows unbounded (1.5 -> 2.5 -> ...)
        # into tanh saturation, then crashes to ~0 around step 1500 and never recovers (all codes look
        # identical at var=0, so no gradient). We had only a FLOOR, no ceiling. Penalize variance ABOVE
        # var_max to hold z in tanh's stable range. LAM_VAR_MAX=0 (default) => off, original behavior.
        self.var_max = float(os.environ.get('LAM_VAR_MAX', 0.0))
        self.var_ceil_lambda = float(os.environ.get('LAM_VAR_CEIL_LAMBDA', 1.0))
        # re-enabled anti-collapse entropy loss (was commented out). Fixed action-code
        # collapse on the tiny model (entropy 0.11 -> 1.75); var penalty alone is not enough.
        self.entropy_lambda = float(os.environ.get('LAM_ENTROPY_LAMBDA', 0.1))
        self.entropy_tau = 0.1  # temperature for soft bin assignment

    def _code_entropy_loss(self, action_latents):
        # MAGVIT-v2 entropy loss on the JOINT code distribution:  L = E[H(p)] - H[E(p)]
        #   term 1 (min): avg per-sample entropy -> each sample commits to one code
        #   term 2 (max): entropy of the batch-avg distribution -> all codes used
        # NOTE: this operates on the full joint distribution over num_bins**action_dim codes,
        # NOT per-dimension marginals. The per-dim version cannot see dimension redundancy:
        # if the two action dims are perfect copies (only codes 00,11 fire) each dim still
        # has max marginal entropy, so the term is satisfied while only 2 of 4 codes are used.
        # The joint batch term H[E(p)] is low when the realized codes concentrate on a subset,
        # so it directly penalizes that redundancy.
        # action_latents: [B, T-1, A]
        # per-dim soft bin assignment (replicate FSQ scale_and_shift(tanh(z)) into [0, num_bins-1])
        bounded_z = 0.5 * (torch.tanh(action_latents) + 1) * (self.quantizer.num_bins - 1)  # [B, T-1, A]
        centers = torch.arange(self.quantizer.num_bins, device=action_latents.device, dtype=action_latents.dtype)  # [num_bins]
        dists_sq = (bounded_z.unsqueeze(-1) - centers) ** 2  # [B, T-1, A, num_bins]
        probs = F.softmax(-dists_sq / self.entropy_tau, dim=-1)  # [B, T-1, A, num_bins]
        # build the joint distribution over all codes via outer product across the A dims
        A = probs.shape[2]
        joint = probs[:, :, 0, :]  # [B, T-1, num_bins]
        for a in range(1, A):
            joint = (joint.unsqueeze(-1) * probs[:, :, a, :].unsqueeze(-2)).flatten(-2)  # [B, T-1, num_bins**(a+1)]
        # joint: [B, T-1, codebook_size]
        log_joint = (joint + 1e-12).log()
        per_sample_entropy = -(joint * log_joint).sum(dim=-1).mean()        # E[H(p)]  -> minimize
        avg_probs = joint.mean(dim=(0, 1))                                  # [codebook_size]
        batch_entropy = -(avg_probs * (avg_probs + 1e-12).log()).sum()      # H[E(p)]  -> maximize
        return per_sample_entropy - batch_entropy

    def forward(self, frames):
        # frames: [B, T, C, H, W]

        # get quantized action latents
        action_latents = self.encoder(frames) # [B, T - 1, A]
        action_latents_quantized = self.quantizer(action_latents) # [B, T - 1, A]

        # decode to get predicted frames
        pred_frames = self.decoder(frames, action_latents_quantized, training=True)  # [B, T - 1, C, H, W]

        # variance of the pre-quant encoder outputs (used for the var penalty AND to gate motion weighting)
        z_var = action_latents.var(dim=0, unbiased=False).mean()

        # reconstruction loss
        target_frames = frames[:, 1:]  # All frames except first [B, T - 1, C, H, W]
        recon_err = F.smooth_l1_loss(pred_frames, target_frames, reduction='none')  # [B,T-1,C,H,W]
        # MOTION-WEIGHTED loss: a uniform pixel loss is dominated by the large static background, so the
        # small moving character barely affects it and the action never learns motion (static recon, weak
        # control). Up-weight pixels that CHANGED between frames. SELF-GATED on encoder variance: motion
        # weighting only turns on once the encoder is healthy (z_var > gate), so it can't DRIVE the variance
        # collapse -- if variance dips, motion backs off and the plain loss + entropy recover it.
        # env-gated; LAM_MOTION_LAMBDA=0 (default) => off.
        motion_lambda = float(os.environ.get('LAM_MOTION_LAMBDA', 0.0))
        motion_gate = float(os.environ.get('LAM_MOTION_GATE', 0.3))  # min z_var to enable motion weighting
        if motion_lambda > 0.0 and z_var.detach().item() > motion_gate:
            motion = (frames[:, 1:] - frames[:, :-1]).abs()  # [B,T-1,C,H,W] where the scene moved
            weight = 1.0 + motion_lambda * motion
            recon_loss = (recon_err * weight).mean()
        else:
            recon_loss = recon_err.mean()

        # variance penalty (helps prevent action collapse) -- FLOOR
        var_penalty = F.relu(self.var_target - z_var)
        # variance CEILING -- caps the runaway into tanh saturation that precedes the collapse
        var_ceiling = F.relu(z_var - self.var_max) if self.var_max > 0.0 else 0.0 * z_var

        # code entropy loss: encourages each sample to commit to one code and all codes to be used
        entropy_loss = self._code_entropy_loss(action_latents)

        total_loss = (recon_loss + self.var_lambda * var_penalty
                      + self.var_ceil_lambda * var_ceiling
                      + self.entropy_lambda * entropy_loss)

        return total_loss, pred_frames

    def encode(self, frames):
        action_latents = self.encoder(frames)  # [B, T, A]
        action_latents_quantized = self.quantizer(action_latents) # [B, T, A]
        return action_latents_quantized
    
    @property
    def model_type(self) -> str:
        return ModelType.LatentActionModel