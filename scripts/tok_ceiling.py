"""Standalone tokenizer reconstruction-ceiling check: tokenize->detokenize->PSNR on held-out Zelda."""
import argparse, torch
from datasets.data_utils import load_data_and_data_loaders
from utils.utils import load_videotokenizer_from_checkpoint
def to01(x): return ((x+1)/2).clamp(0,1)
def psnr(a,b):
    mse=((to01(a)-to01(b))**2).flatten(1).mean(1).clamp_min(1e-12); return 10*torch.log10(1/mse)
p=argparse.ArgumentParser(); p.add_argument('--ckpt',required=True); p.add_argument('--n',type=int,default=12); p.add_argument('--bs',type=int,default=8)
a=p.parse_args(); dev='cpu'; torch.set_grad_enabled(False)
tok,_=load_videotokenizer_from_checkpoint(a.ckpt,dev); tok.eval()
_,_,_,loader,_=load_data_and_data_loaders(dataset='ZELDA',batch_size=a.bs,num_frames=4,preload_ratio=0.15)
tot=0.0; n=0; it=iter(loader)
for _ in range(a.n):
    try: b=next(it)
    except StopIteration: break
    f=(b[0] if isinstance(b,(list,tuple)) else b).to(dev)
    rec=tok.detokenize(tok.quantizer.get_latents_from_indices(tok.tokenize(f)))
    for t in range(f.shape[1]):
        tot+=psnr(rec[:,t],f[:,t]).sum().item(); n+=f.shape[0]
print(f"LARGE tokenizer recon ceiling: {tot/n:.2f} dB  ({n} frames)  [tiny tokenizer was ~32.5]")
