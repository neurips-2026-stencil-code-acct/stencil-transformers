import os, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",   type=str, default="checkpoints/alpha_0.4/best_model.pt")
parser.add_argument("--data_dir",     type=str, default="data/alpha_0.4")
parser.add_argument("--stencil_path", type=str, default="data/alpha_0.4/stencil.npy")
parser.add_argument("--save_dir",     type=str, default="results/alpha_0.4")
parser.add_argument("--n_batches",    type=int, default=200)
parser.add_argument("--batch_size",   type=int, default=256)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print(f"\nLoading checkpoint: {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
saved_args = ckpt["args"]
NX       = 64
N_LAYERS = saved_args.get("n_layers", 4)
N_HEADS  = saved_args.get("n_heads",  4)
D_MODEL  = saved_args.get("d_model",  64)
D_FF     = saved_args.get("d_ff",     256)
DROPOUT  = saved_args.get("dropout",  0.1)
print(f"Model: {N_LAYERS}L x {N_HEADS}H x d{D_MODEL}")
print(f"Epoch: {ckpt['epoch']} | val MSE: {ckpt['val_loss']:.2e}")

class SpatialEmbedding(nn.Module):
    def __init__(self, nx, d_model):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        pe = torch.zeros(nx, d_model)
        pos = torch.arange(nx).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.value_proj(x)
        return x + self.pe

class VanillaTransformer(nn.Module):
    def __init__(self, nx, n_layers, n_heads, d_model, d_ff, dropout):
        super().__init__()
        self.embedding = SpatialEmbedding(nx, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output_head(x).squeeze(-1)

model = VanillaTransformer(NX, N_LAYERS, N_HEADS, D_MODEL, D_FF, DROPOUT).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("Model loaded.")

def collect_attn(mdl, loader, n_batches, device):
    """
    Memory-optimized: accumulates a running SUM of attention weights per
    (layer, head) instead of storing every individual batch's full attention
    tensor. Original version stored every batch (list of (B,H,NX,NX) tensors,
    concatenated at the end) -- for n_batches=200, batch_size=256, 4 layers,
    that's ~12.5GB per call, which OOMs when multiple metrics.py processes run
    in parallel (e.g. one per GPU slot in a multi-seed sweep). This version
    keeps peak memory at O(n_layers * n_heads * NX * NX) regardless of
    n_batches -- a few hundred KB instead of gigabytes.

    Returns: {layer_idx: (n_heads, NX, NX) mean attention} -- already averaged
    over all examples, NOT the raw (n_examples, n_heads, NX, NX) the original
    returned. Downstream code that did `attn_data[li][:, hi].mean(0)` should
    now just index `attn_data[li][hi]` directly.
    """
    sums = {i: torch.zeros(N_HEADS, NX, NX) for i in range(N_LAYERS)}
    counts = {i: 0 for i in range(N_LAYERS)}

    def make_hook(idx):
        def hook(module, inp, out):
            with torch.no_grad():
                # norm_first=True: self_attn actually runs on norm1(src) during
                # the real forward pass, not on the raw layer input.
                normed = module.norm1(inp[0])
                _, w = module.self_attn(normed, normed, normed,
                    need_weights=True, average_attn_weights=False)
                w = w.detach().cpu()  # (B, H, NX, NX)
                sums[idx] += w.sum(dim=0)
                counts[idx] += w.shape[0]
        return hook

    hooks = [layer.register_forward_hook(make_hook(i))
             for i, layer in enumerate(mdl.transformer.layers)]
    mdl.eval()
    with torch.no_grad():
        for bi, x in enumerate(loader):
            if bi >= n_batches: break
            _ = mdl(x.to(device))
            if (bi+1) % 10 == 0: print(f"  Batch {bi+1}/{n_batches}")
    for h in hooks: h.remove()

    return {i: (sums[i] / counts[i]).numpy() if counts[i] > 0 else None
            for i in range(N_LAYERS)}

print("\nLoading test data...")
test_data = np.load(os.path.join(args.data_dir, "test.npy"))
inputs = torch.tensor(test_data[:, :-1, :].reshape(-1, 64), dtype=torch.float32)
loader = DataLoader(inputs, batch_size=args.batch_size, shuffle=False)
print(f"Collecting attention ({args.n_batches} batches)...")
attn_data = collect_attn(model, loader, args.n_batches, device)
print("Done.")

stencil = np.load(args.stencil_path)
r_val = stencil[0]
print(f"\nStencil: {stencil}  r={r_val:.4f}")

def extract_nb(attn, nx):
    nb = np.zeros((nx, 3))
    for i in range(nx):
        nb[i] = [attn[i,(i-1)%nx], attn[i,i], attn[i,(i+1)%nx]]
    return nb

def corr(attn, stencil, nx):
    mean = extract_nb(attn, nx).mean(0)
    if np.std(mean) < 1e-10 or np.std(stencil) < 1e-10: return 0.0
    return float(pearsonr(mean, stencil)[0])

def locality(attn, nx):
    return float(extract_nb(attn, nx).sum(1).mean())

def entropy(attn):
    a = np.clip(attn, 1e-10, 1.0)
    return float(-(a * np.log(a)).sum(1).mean())

# Baselines
# Per-head metrics
print("\nPer-head metrics:")
C = np.zeros((N_LAYERS,N_HEADS))
L = np.zeros((N_LAYERS,N_HEADS))
E = np.zeros((N_LAYERS,N_HEADS))
store = {}
for li in range(N_LAYERS):
    if attn_data[li] is None: continue
    for hi in range(N_HEADS):
        ma = attn_data[li][hi]  # already the per-head mean -- collect_attn now
                                 # returns pre-averaged (n_heads, NX, NX) arrays,
                                 # so no further [:, hi].mean(0) needed here
        store[(li,hi)] = ma
        C[li,hi] = corr(ma,stencil,NX)
        L[li,hi] = locality(ma,NX)
        E[li,hi] = entropy(ma)
    print(f"  Layer {li+1}: corr={np.round(C[li],3)}")

np.save(os.path.join(args.save_dir,"corr_matrix.npy"), C)
np.save(os.path.join(args.save_dir,"locality_matrix.npy"), L)
np.save(os.path.join(args.save_dir,"entropy_matrix.npy"), E)
mean_attn_array = np.stack([store[(li,hi)] 
                             for li in range(N_LAYERS) 
                             for hi in range(N_HEADS)])
np.save(os.path.join(args.save_dir, "mean_attn.npy"), mean_attn_array)
print(f"\nSaved to {args.save_dir}/")

bl,bh = divmod(np.argmax(C),N_HEADS)
wl,wh = divmod(np.argmin(C),N_HEADS)

print("\n"+"="*50)
print(f"Best:  L{bl+1}H{bh+1} = {C[bl,bh]:.4f}")
print(f"Worst: L{wl+1}H{wh+1} = {C[wl,wh]:.4f}")
print(f"Mean:  {C.mean():.4f} +/- {C.std():.4f}")
print(f"Locality mean: {L.mean():.4f}")
print(f"Entropy mean:  {E.mean():.4f}")
print("="*50)