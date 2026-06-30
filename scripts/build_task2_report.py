"""Task 2 report: enhancing the TinyWorlds architecture.
Generates an illustrated PDF (diagrams + bullet ideology) with matplotlib PdfPages.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import os

INK="#1a1a2e"; BLUE="#2d6cdf"; TEAL="#179a8f"; ORANGE="#e8833a"; RED="#d4453b"
GREY="#9aa0b4"; LGREY="#e8eaf2"; GREEN="#2e9e5b"; PURPLE="#7a4fd0"; YELLOW="#f3c14b"

def box(ax,x,y,w,h,text,fc="white",ec=INK,fs=9,tc=INK,lw=1.4,style="round,pad=0.02",bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=style,fc=fc,ec=ec,lw=lw,zorder=2))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color=tc,zorder=3,
            fontweight="bold" if bold else "normal",wrap=True)

def arrow(ax,x1,y1,x2,y2,c=INK,lw=1.6,style="-|>",ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=14,
                color=c,lw=lw,linestyle=ls,zorder=1))

def bullets(ax,x,y,items,fs=10.5,dy=0.052,head=None,hc=BLUE):
    if head:
        ax.text(x,y,head,fontsize=fs+2.5,fontweight="bold",color=hc,va="top");y-=dy*1.5
    for it in items:
        lvl=it[0]; txt=it[1]; col=it[2] if len(it)>2 else INK
        mk={0:"●",1:"–",2:"·"}[lvl]
        ax.text(x+lvl*0.03,y,mk,fontsize=fs-1,color=hc if lvl==0 else GREY,va="top")
        ax.text(x+lvl*0.03+0.022,y,txt,fontsize=fs-(lvl*0.5),color=col,va="top",
                wrap=True)
        y-=dy*(1.0+0.12*txt.count("\n"))
    return y

def page(pp,draw):
    fig=plt.figure(figsize=(8.5,11)); ax=fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    draw(ax)
    pp.savefig(fig); plt.close(fig)

def header(ax,kicker,title):
    ax.add_patch(Rectangle((0,0.945),1,0.055,fc=INK,ec="none"))
    ax.text(0.06,0.972,title,fontsize=17,fontweight="bold",color="white",va="center")
    ax.text(0.06,0.93,kicker,fontsize=10,color=BLUE,va="center",style="italic")

# ---------------- PAGE 1: COVER ----------------
def p_cover(ax):
    ax.add_patch(Rectangle((0,0),1,1,fc=INK,ec="none"))
    ax.text(0.5,0.74,"Enhancing the TinyWorlds\nNetwork Architecture",ha="center",
            fontsize=27,fontweight="bold",color="white",linespacing=1.3)
    ax.add_patch(Rectangle((0.2,0.69),0.6,0.004,fc=BLUE,ec="none"))
    ax.text(0.5,0.63,"Task 2  ·  Architecture research track",ha="center",fontsize=13,color=TEAL)
    # mini diagram: attention block -> mamba block
    box(ax,0.18,0.42,0.26,0.10,"Current\nTransformer block",fc=BLUE,tc="white",ec="white",fs=11,bold=True)
    arrow(ax,0.46,0.47,0.56,0.47,c="white",lw=2.2)
    box(ax,0.57,0.42,0.26,0.10,"Mamba / SSM\n+ compression",fc=TEAL,tc="white",ec="white",fs=11,bold=True)
    items=[
        (0,"What the current network looks like (the shared backbone)","white"),
        (0,"Which layer we target, and what we replace it with","white"),
        (0,"The techniques in detail: replacement, pruning, quantization","white"),
        (0,"The methodology and how we measure success","white"),
    ]
    bullets(ax,0.2,0.34,items,fs=11.5,dy=0.05,hc=YELLOW)
    ax.text(0.5,0.06,"Goal: same quality at lower cost — better architecture per parameter, not more parameters",
            ha="center",fontsize=10.5,color=GREY,style="italic")

# ---------------- PAGE 2: CURRENT ARCHITECTURE ----------------
def p_current(ax):
    header(ax,"Where we are","1.  The current architecture")
    bullets(ax,0.06,0.90,[
        (0,"One shared backbone (the Space-Time Transformer, STTransformer) powers all three"),
        (1,"models: the video tokenizer, the latent action model, and the dynamics model."),
        (0,"It is a stack of N identical blocks. Each block is shape-preserving:  [B, T, P, E] in → [B, T, P, E] out."),
        (0,"Large dynamics config:  embed=512, heads=8, hidden=2048, N=18 blocks."),
        (0,"Every block has three stages, each wrapped with a residual + a conditioned norm (FiLM on the action):"),
    ],fs=10.5,dy=0.043)
    # left: the stack
    ax.text(0.20,0.585,"The stack (N=18)",ha="center",fontsize=10,fontweight="bold",color=INK)
    y=0.55
    for i in range(5):
        lab=["Block 1  (touches input embed)","Block 2","⋮   middle blocks   ⋮","Block 17","Block 18  (touches output head)"][i]
        fc=LGREY if i in (0,4) else "white"; ec=GREY if i in (0,4) else INK
        box(ax,0.07,y,0.26,0.05,lab,fc=fc,ec=ec,fs=8.2); y-=0.062
    arrow(ax,0.20,0.55,0.20,0.235,c=GREY,lw=1.2,ls=(0,(3,3)))
    # right: one block exploded
    bx=0.46; ax.text(bx+0.24,0.585,"Inside one block",ha="center",fontsize=10,fontweight="bold",color=INK)
    box(ax,bx,0.50,0.48,0.055,"Spatial Attention   (over P patches, within a frame)",fc="#e8f0ff",ec=BLUE,fs=8.6)
    arrow(ax,bx+0.24,0.50,bx+0.24,0.47)
    box(ax,bx,0.415,0.48,0.055,"Temporal Attention   (over T frames, causal)",fc="#ffe9d6",ec=ORANGE,fs=8.6,bold=True)
    arrow(ax,bx+0.24,0.415,bx+0.24,0.385)
    box(ax,bx,0.33,0.48,0.055,"SwiGLU Feed-Forward   (per-token MLP)",fc="#e7f7ef",ec=GREEN,fs=8.6)
    ax.text(bx+0.24,0.30,"each stage:  x ← Norm( x + Stage(x) , action )",ha="center",fontsize=8.2,color=GREY,style="italic")
    ax.text(0.345,0.4425,"TARGET →",ha="right",va="center",fontsize=9,color=RED,fontweight="bold")
    bullets(ax,0.06,0.225,[
        (0,"Spatial Attention mixes the patches inside each frame (full attention, fine to keep)."),
        (0,"Temporal Attention mixes information across time at each patch position — it is causal and is the",RED),
        (1,"expensive sequence mixer:  cost grows with T² (every frame attends to every other frame).",RED),
        (0,"SwiGLU FFN is a standard position-wise MLP."),
    ],fs=10,dy=0.045)

# ---------------- PAGE 3: TARGET LAYER ----------------
def p_target(ax):
    header(ax,"What we change","2.  The layer we target:  Temporal Attention")
    bullets(ax,0.06,0.90,[
        (0,"We replace the Temporal Attention sub-layer (the over-time mixer), not the whole block."),
        (0,"Why this layer: it is the sequence model over frames — exactly where cost and long-context limits live."),
        (0,"Spatial attention stays: patches-within-a-frame is a small fixed set and full attention is cheap there."),
    ],fs=10.5,dy=0.045)
    # cost diagram: T x T attention matrix
    ax.text(0.27,0.745,"Temporal Attention  =  T × T interaction",ha="center",fontsize=10,fontweight="bold",color=ORANGE)
    n=6; x0,y0,c=0.12,0.42,0.046
    for i in range(n):
        for j in range(n):
            fc=ORANGE if j<=i else LGREY  # causal lower-triangular
            ax.add_patch(Rectangle((x0+j*c,y0+(n-1-i)*c),c*0.92,c*0.92,fc=fc,ec="white",lw=1))
    ax.text(x0+n*c/2,y0-0.04,"frame j (key)",ha="center",fontsize=8,color=GREY)
    ax.text(x0-0.03,y0+n*c/2,"frame i (query)",ha="center",fontsize=8,color=GREY,rotation=90)
    ax.text(x0+n*c/2,y0+n*c+0.025,"every frame compares to every earlier frame",ha="center",fontsize=8.5,color=ORANGE)
    ax.text(0.27,0.135,"Cost & memory scale as  O(T²)",ha="center",fontsize=11,color=RED,fontweight="bold")
    # right: why it matters
    bullets(ax,0.55,0.74,[
        (0,"Quadratic in the number of frames T.",RED),
        (1,"Doubling context (4→16 frames) is far more"),
        (2,"than 4× the temporal cost."),
        (0,"This caps how much history the model can"),
        (1,"afford — and our diagnosis showed longer"),
        (1,"context is exactly what a stronger model wants."),
        (0,"It is causal already (frame i sees only ≤ i),"),
        (1,"which makes it a clean match for a recurrent"),
        (1,"replacement that is causal by construction."),
        (0,"Start with ONE middle block (block 9 of 18):",PURPLE),
        (1,"middle blocks are pure [B,T,P,E]→[B,T,P,E],",PURPLE),
        (1,"so a drop-in swap is easy to verify; the first",PURPLE),
        (1,"and last blocks touch embeddings / heads.",PURPLE),
    ],fs=9.8,dy=0.042)

# ---------------- PAGE 4: REPLACEMENT ----------------
def p_replace(ax):
    header(ax,"What we replace it with","3.  The replacement:  Mamba / State-Space block")
    bullets(ax,0.06,0.90,[
        (0,"Replace the T×T temporal attention with a selective State-Space Model (Mamba)."),
        (0,"Same contract:  reshape [B,T,P,E] → [(B·P), T, E], scan over time, reshape back, residual + norm."),
    ],fs=10.5,dy=0.045)
    # left: attention (quadratic)
    ax.text(0.27,0.80,"Attention  —  O(T²)",ha="center",fontsize=10.5,fontweight="bold",color=ORANGE)
    n=5;x0,y0,c=0.13,0.55,0.05
    for i in range(n):
        for j in range(n):
            fc=ORANGE if j<=i else LGREY
            ax.add_patch(Rectangle((x0+j*c,y0+(n-1-i)*c),c*0.9,c*0.9,fc=fc,ec="white"))
    ax.text(0.27,0.52,"stores a frame-by-frame matrix",ha="center",fontsize=8.3,color=GREY)
    # right: mamba (linear recurrence)
    ax.text(0.74,0.80,"Mamba / SSM  —  O(T)",ha="center",fontsize=10.5,fontweight="bold",color=TEAL)
    xs=[0.60,0.68,0.76,0.84]; yb=0.66
    for k,xx in enumerate(xs):
        box(ax,xx,yb,0.06,0.05,f"x{k+1}",fc="#eafaf6",ec=TEAL,fs=8)
        if k>0: arrow(ax,xs[k-1]+0.06,yb-0.03,xx+0.03,yb-0.005,c=TEAL,lw=1.3)
        box(ax,xx,yb-0.10,0.06,0.05,f"h{k+1}",fc=TEAL,tc="white",ec=TEAL,fs=8)
        arrow(ax,xx+0.03,yb,xx+0.03,yb-0.05,c=GREY,lw=1.1)
    ax.text(0.74,yb-0.135,"carries a running state, frame by frame",ha="center",fontsize=8.3,color=GREY)
    ax.text(0.74,0.48,"h_t = A·h_{t-1} + B·x_t\n y_t = C·h_t + D·x_t",ha="center",fontsize=9.5,color=INK,
            family="monospace",linespacing=1.5)
    bullets(ax,0.06,0.42,[
        (0,"How it works:  it keeps a running hidden state and updates it one frame at a time (a recurrence),"),
        (1,"instead of comparing all frame pairs. “Selective” = the update gates (A, B, C) depend on the input,"),
        (1,"so it can choose what to remember and what to forget — this is what makes Mamba competitive with attention."),
        (0,"Linear in T and constant memory in the state → long context (16, 32, 64 frames) becomes affordable.",TEAL),
        (0,"Causal by construction (state only flows forward), matching the temporal layer it replaces.",TEAL),
        (0,"Parameter-matched:  we size the SSM block to the same parameter count as the attention block,"),
        (1,"so any change in quality is attributable to the architecture, not to more parameters (the prof’s constraint)."),
        (0,"If Mamba underperforms, the same slot accepts other linear-cost mixers: linear attention, RetNet, RWKV.",GREY),
    ],fs=9.8,dy=0.0425)

# ---------------- PAGE 5: TECHNIQUES ----------------
def p_techniques(ax):
    header(ax,"The toolbox in detail","4.  Techniques")
    def band(y,text,c):
        ax.add_patch(Rectangle((0.06,y),0.88,0.046,fc=c,ec="none"))
        ax.text(0.5,y+0.023,text,ha="center",va="center",fontsize=11.5,fontweight="bold",color="white")
    # A (full width, no diagram)
    band(0.87,"A.   Architecture replacement  (Mamba / SSM)",TEAL)
    bullets(ax,0.08,0.84,[
        (0,"Swap temporal attention → SSM block; start middle, expand to more blocks if it holds."),
        (0,"Benefit: linear-cost long context, O(T²) → O(T), at a matched parameter count."),
    ],fs=9.8,dy=0.04)
    # B pruning  (bullets left, diagram right)
    band(0.72,"B.   Network Slimming  (structured pruning)",PURPLE)
    bullets(ax,0.08,0.69,[
        (0,"L1 penalty on each channel’s norm scale γ"),
        (1,"during training → unused channels → γ ≈ 0."),
        (0,"Remove the low-γ channels, then fine-tune."),
        (0,"Smaller & faster, almost no quality loss.",PURPLE),
    ],fs=9.6,dy=0.039)
    xs=[0.635,0.683,0.731,0.779,0.827,0.875]; g=[0.95,0.12,0.8,0.06,0.6,0.15]; yb=0.635
    for xx,gv in zip(xs,g):
        h=0.018+gv*0.05; pruned=gv<0.2
        ax.add_patch(Rectangle((xx,yb),0.038,h,fc=GREY if pruned else PURPLE,ec="white",lw=1))
        if pruned: ax.text(xx+0.019,yb+h+0.006,"✕",ha="center",fontsize=8,color=RED)
    ax.text(0.755,0.62,"tall = kept   ·   ✕ = pruned",ha="center",fontsize=7.8,color=GREY)
    # C quantization  (bullets left, diagram right)
    band(0.50,"C.   Quantization  (precision reduction)",BLUE)
    bullets(ax,0.08,0.47,[
        (0,"Store / compute in INT8 instead of FP32"),
        (1,"→ about 4× smaller and faster."),
        (0,"Post-training (calibrate) or QA-training."),
        (0,"Existing libraries do it; ask Jiacheng.",GREY),
    ],fs=9.6,dy=0.039)
    box(ax,0.63,0.415,0.095,0.05,"FP32\n32 bits",fc="#e8f0ff",ec=BLUE,fs=8.5)
    arrow(ax,0.73,0.44,0.785,0.44,c=BLUE,lw=2.2)
    box(ax,0.795,0.423,0.10,0.034,"INT8 · 8 bits",fc=BLUE,tc="white",ec="none",fs=8.2)
    ax.text(0.79,0.40,"4× smaller",ha="center",fontsize=7.8,color=GREY)
    # D others (full width)
    band(0.29,"D.   Also on the table",GREY)
    bullets(ax,0.08,0.26,[
        (0,"Knowledge distillation: a small student imitates the large model’s outputs."),
        (0,"Low-rank factorization / weight sharing across blocks: fewer parameters, similar quality."),
    ],fs=9.8,dy=0.04)
    ax.text(0.5,0.11,"Replacement changes the architecture;  pruning & quantization shrink whatever architecture we keep.",
            ha="center",fontsize=9.5,color=INK,style="italic")

# ---------------- PAGE 6: METHODOLOGY ----------------
def p_method(ax):
    header(ax,"How we proceed","5.  Methodology  &  how we measure")
    ax.text(0.06,0.90,"The ideology",fontsize=13,fontweight="bold",color=BLUE,va="top")
    steps=[("1","Replace ONE middle block’s temporal mixer with a Mamba/SSM block.",TEAL),
           ("2","Verify correctness: shapes match, it trains, loss tracks the all-attention baseline.",TEAL),
           ("3","Expand: 2 blocks → more → eventually all temporal mixers.",TEAL),
           ("4","Then compress the resulting network: Network Slimming, then quantization.",PURPLE),
           ("5","Re-evaluate at every step against the baseline.",BLUE)]
    y=0.84
    for n,t,c in steps:
        ax.add_patch(Circle((0.10,y),0.018,fc=c,ec="none"))
        ax.text(0.10,y,n,ha="center",va="center",fontsize=9.5,color="white",fontweight="bold")
        ax.text(0.14,y,t,va="center",fontsize=10.3,color=INK)
        if n!="5": arrow(ax,0.10,y-0.018,0.10,y-0.052,c=GREY,lw=1.2)
        y-=0.07
    ax.text(0.06,0.44,"Guardrails",fontsize=13,fontweight="bold",color=BLUE,va="top")
    bullets(ax,0.06,0.40,[
        (0,"Matched parameter count throughout — the prof’s constraint; isolates architecture from scale."),
        (0,"Start in the middle — cleanest interface, fastest correctness check before touching ends."),
        (0,"One change at a time — so every gain or loss is attributable."),
    ],fs=10.2,dy=0.05)
    ax.text(0.06,0.235,"What we report",fontsize=13,fontweight="bold",color=BLUE,va="top")
    bullets(ax,0.06,0.195,[
        (0,"Quality:  single-step PSNR / SSIM and rollout PSNR vs the attention baseline."),
        (0,"Efficiency:  parameter count, FLOPs, peak memory, and step / inference latency."),
        (0,"The win condition:  match the baseline’s quality at lower cost — efficiency per parameter."),
    ],fs=10.2,dy=0.05)

def main():
    out=os.path.join(os.path.expanduser("~"),"TinyWorlds_task2_architecture_report.pdf")
    with PdfPages(out) as pp:
        for d in [p_cover,p_current,p_target,p_replace,p_techniques,p_method]:
            page(pp,d)
    print("WROTE",out)

if __name__=="__main__":
    main()
