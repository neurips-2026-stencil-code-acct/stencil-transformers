"""
transformer.py
Vanilla spatiotemporal transformer for heat equation next-step prediction.
No physics constraints. Trains on synthetic FD data from generate.py.

Usage:
    python transformer.py # local, default params
    python transformer.py --epochs 200 # more epochs
    python transformer.py --data_dir /path/data # custom data path
    python transformer.py --epochs 90 --resume_from checkpoints/.../resume_state.pt  # continue training

IMPORTANT: --epochs is always the FINAL TARGET total epoch count, fixed for
the whole run (including all resumes), because the cosine LR schedule's
T_max is set from it. Do not change --epochs between resumes of the same run.
"""
"""
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False <- INCLUDE
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#============ CLI args ============
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--save_dir", type=str, default="checkpoints")
parser.add_argument("--epochs", type=int, default=2,
                     help="FINAL TARGET total epochs (fixes cosine LR T_max). "
                          "Must stay the same across resumes of the same run.")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--resume_from", type=str, default=None,
                     help="Path to a resume_state.pt checkpoint to continue training from.")

# for saving purposes 
parser.add_argument("--alpha", type=float, default=0.25)

#============ Model Parameters ============
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--d_ff", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.1)
args = parser.parse_args()

#============ Reproducibility ============
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#============ Device ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ======================================================
# 1. DATASET
# ======================================================
class HeatDataset(Dataset):
    """
    Each sample is one (input, target) pair from a trajectory.
    Input: u[t] - spatial snapshot at time t, shape (NX,)
    Target: u[t+1] - next timestep, shape (NX,)

    We flatten all timestep pairs across all trajectories,
    giving N_trajectories * NT total samples.
    """
    def __init__(self, path, max_samples=None):
        data = np.load(path) # (N, NT+1, NX)
        # All consecutive pairs across all trajectories
        inputs  = data[:, :-1, :] # (N, NT, NX)
        targets = data[:, 1:,  :] # (N, NT, NX)
        N, T, X = inputs.shape

        # Flatten trajectory and time dimensions -> (N*T, NX)

        self.inputs = torch.tensor(
            inputs.reshape(N * T, X), dtype=torch.float32)
        self.targets = torch.tensor(
            targets.reshape(N * T, X), dtype=torch.float32)
        
        if max_samples is not None:
            self.inputs  = self.inputs[:max_samples]
            self.targets = self.targets[:max_samples]
        print(f"Loaded {path}: {N} trajectories x {T} steps = {len(self)} samples")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

print("\nLoading datasets...")
train_ds = HeatDataset(os.path.join(args.data_dir, "train.npy"), max_samples=args.max_samples)
val_ds = HeatDataset(os.path.join(args.data_dir, "val.npy"))

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0, pin_memory=True)

val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                          shuffle=False, num_workers=0, pin_memory=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL
# ─────────────────────────────────────────────────────────────────────────────
class SpatialEmbedding(nn.Module):
    """
    Embeds each spatial token (scalar temperature value) into d_model dims.
    Each of the NX spatial positions becomes one token.
    Adds sinusoidal positional encoding so the model knows spatial order.

    Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
    """
    def __init__(self, nx, d_model):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model) # scalar -> d_model
        # Sinusoidal positional encoding - fixed, not learned
        # Shape: (1, NX, d_model)
        pe = torch.zeros(nx, d_model)
        pos = torch.arange(nx).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0)) # (1, NX, d_model)

    def forward(self, x):
        # x: (B, NX) - one scalar per spatial token
        x = x.unsqueeze(-1) # (B, NX, 1)
        x = self.value_proj(x) # (B, NX, d_model)
        x = x + self.pe # add positional encoding
        return x # (B, NX, d_model)


class VanillaTransformer(nn.Module):
    """
    Vanilla spatiotemporal transformer for next-step PDE prediction.
    No physics constraints. Standard encoder-only architecture.

    Architecture:
        NX spatial tokens -> embedding -> L transformer layers -> linear head -> NX outputs

    Attention weights are extracted post-hoc for stencil emergence analysis.
    """
    def __init__(self, nx, n_layers, n_heads, d_model, d_ff, dropout):
        super().__init__()
        self.embedding = SpatialEmbedding(nx, d_model)

        # Standard transformer encoder layers
        """
        Pre-LayerNorm (norm_first=True) - using pre-LN instead of post-LN is a specific architectural choice from:
        Xiong et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.
        """
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True, # (B, seq, d_model) convention

            norm_first=True, # Pre-LN - more stable training
        )
        """
        [Paszke et al., 2019]
        """
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        # Output head: d_model -> 1 scalar per spatial token
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, NX)
        x = self.embedding(x) # (B, NX, d_model)
        x = self.transformer(x) # (B, NX, d_model)
        x = self.output_head(x) # (B, NX, 1)
        return x.squeeze(-1) # (B, NX)


# ─────────────────────────────────────────────────────────────────────────────
# NOTE ON ATTENTION EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
# PyTorch's TransformerEncoderLayer does not expose attention weights by
# default. We register forward hooks on each layer's self_attn module to
# capture them during the analysis phase. We don't need them during
# training - only when running metrics.py after training is complete.
# ─────────────────────────────────────────────────────────────────────────────

NX = train_ds.inputs.shape[1]
model = VanillaTransformer(
    nx = NX,
    n_layers = args.n_layers,
    n_heads = args.n_heads,
    d_model  = args.d_model,
    d_ff = args.d_ff,
    dropout = args.dropout,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: {args.n_layers}L × {args.n_heads}H × d{args.d_model}")
print(f"Parameters: {n_params:,}")
# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING
# ─────────────────────────────────────────────────────────────────────────────
criterion = nn.MSELoss()
"""
Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization." ICLR.
"""
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Cosine annealing - reduces LR smoothly to near zero over training
# T_max is fixed to args.epochs (the FINAL target), set once and never
# changed across resumes - this keeps the LR trajectory coherent whether
# the run happens in one shot or across several resumed sessions.
"""
Cosine annealing scheduler - from:
Loshchilov & Hutter (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts." ICLR.
"""
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-5
)

os.makedirs(args.save_dir, exist_ok=True)

best_val_loss = float("inf")
train_losses = []
val_losses = []
epoch_times = []
start_epoch = 1

# ── Resume from checkpoint, if provided ────────────────────────────────────
if args.resume_from is not None:
    print(f"\nResuming from checkpoint: {args.resume_from}")
    ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)

    saved_target_epochs = ckpt["args"]["epochs"]
    if saved_target_epochs != args.epochs:
        raise ValueError(
            f"Checkpoint was trained toward a target of {saved_target_epochs} epochs, "
            f"but --epochs={args.epochs} was passed now. These must match, or the "
            f"cosine LR schedule will be inconsistent across the resumed run."
        )

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    train_losses = list(ckpt["train_losses"])
    val_losses = list(ckpt["val_losses"])
    epoch_times = list(ckpt.get("epoch_times", []))
    best_val_loss = ckpt["best_val_loss"]
    start_epoch = ckpt["epoch"] + 1

    print(f"Resumed at epoch {start_epoch} (of {args.epochs} target), "
          f"best val loss so far: {best_val_loss:.4e}")

    if start_epoch > args.epochs:
        print(f"\nCheckpoint epoch ({ckpt['epoch']}) already reached or exceeded "
              f"target ({args.epochs}). Nothing to do.")
        raise SystemExit

print(f"\nTraining epochs {start_epoch} -> {args.epochs} (target {args.epochs})...")
print(f"{'Epoch':>6} {'Train MSE':>12} {'Val MSE':>12} {'LR':>10} {'Time(s)':>10}")
print("-" * 58)

training_start = time.perf_counter()

for epoch in range(start_epoch, args.epochs + 1):
    epoch_start = time.perf_counter()

    # ── Train ─────────────────────────────────────────────────────────────────

    """
    Pascanu et al. (2013). "On the difficulty of training recurrent neural networks." ICML. 
    for Gradient Clipping 
    """
    model.train()
    running_loss = 0.0
    n_batches = len(train_loader)
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if batch_idx % 500 == 0:
            print(f"  Epoch {epoch} | batch {batch_idx}/{n_batches} | loss {loss.item():.12f}", flush=True)
        running_loss += loss.item() * x.size(0)
    train_loss = running_loss / len(train_ds)

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            running_val += criterion(pred, y).item() * x.size(0)
    val_loss = running_val / len(val_ds)

    scheduler.step()
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # ── Timing ────────────────────────────────────────────────────────────────
    epoch_time = time.perf_counter() - epoch_start
    epoch_times.append(epoch_time)

    # ── Logging ───────────────────────────────────────────────────────────────
    lr_now = scheduler.get_last_lr()[0]
    print(f"{epoch:>6} {train_loss:>12.2e} {val_loss:>12.2e} {lr_now:>10.2e} {epoch_time:>10.2f}")

    # ── Checkpoint best model (eval/analysis use - model weights only) ─────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "args": vars(args),
        }, os.path.join(args.save_dir, "best_model.pt"))

    # ── Checkpoint resume state (every epoch - full state for continuing later) ─
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epoch_times": epoch_times,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }, os.path.join(args.save_dir, "resume_state.pt"))

total_time = time.perf_counter() - training_start

# ── Save final model + loss curves ────────────────────────────────────────────
torch.save({
    "epoch": args.epochs,
    "model_state_dict": model.state_dict(),
    "train_losses": train_losses,
    "val_losses": val_losses,
    "args": vars(args),
}, os.path.join(args.save_dir, "final_model.pt"))

np.save(os.path.join(args.save_dir, "train_losses.npy"), np.array(train_losses))
np.save(os.path.join(args.save_dir, "val_losses.npy"), np.array(val_losses))
np.save(os.path.join(args.save_dir, "epoch_times.npy"), np.array(epoch_times))

print(f"\nDone. Best val MSE: {best_val_loss:.6f}")
print(f"This session's training time: {total_time:.1f}s ({total_time/60:.2f} min)")
if epoch_times:
    print(f"Mean epoch time (this session): {np.mean(epoch_times):.2f}s")
print(f"Saved to {args.save_dir}/best_model.pt")
print(f"{args.save_dir}/resume_state.pt  (use with --resume_from to continue)")
print(f"{args.save_dir}/final_model.pt")
print(f"{args.save_dir}/train_losses.npy")