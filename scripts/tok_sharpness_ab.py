import os, torch, torch.nn.functional as F
from models.video_tokenizer import VideoTokenizer
from datasets.data_utils import load_data_and_data_loaders
from utils.utils import load_videotokenizer_from_checkpoint
from utils.metrics import frame_psnr

dev='cpu'
def build(): return VideoTokenizer(frame_size=(64,64), patch_size=4, embed_dim=32, num_heads=8,
                                   hidden_dim=128, num_blocks=4, latent_dim=5, num_bins=4).to(dev)
def lap_sharp(x):  # x:[N,3,H,W] in [-1,1]; mean variance of Laplacian (higher=sharper)
    g = x.mean(1,keepdim=True)
    k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32).view(1,1,3,3)
    return F.conv2d(g,k,padding=1).var(dim=(1,2,3)).mean().item()

ckpts = {'OLD (L1)':os.environ['OLD'], 'NEW (perceptual)':os.environ['NEW']}
_,_,loader,_,_ = load_data_and_data_loaders(dataset='ZELDA', batch_size=12, num_frames=4)
clips=[]
it=iter(loader)
for _ in range(2): clips.append(next(it)[0].to(dev))
gt = torch.cat(clips,0)  # [N,4,3,64,64]
print(f"clips: {gt.shape[0]} | GT sharpness: {lap_sharp(gt.reshape(-1,*gt.shape[2:])):.2f}")
for name,ck in ckpts.items():
    m=build(); m,_=load_videotokenizer_from_checkpoint(ck,dev,m,False); m.eval()
    with torch.no_grad():
        xh = m(gt)[1]
    p = frame_psnr(xh, gt).mean().item()
    s = lap_sharp(xh.reshape(-1,*xh.shape[2:]))
    print(f"{name:18s} -> recon PSNR {p:6.3f} dB | sharpness {s:6.2f}")
