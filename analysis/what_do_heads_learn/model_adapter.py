"""
model_adapter.py

Provides two of extract_attention_profiles.py's three seams (see info.md): the
--model-factory that turns a raw training checkpoint -- a
{"model_state_dict", "args"} dict, not a pickled nn.Module -- into a loaded
VanillaTransformer, and the --attention-extractor that this project's model
actually needs (attention_common.extract_attention's monkeypatches confirmed not to
fire against it; see extract_attention's docstring below).

Why this file exists and not transformer.py: extract_profiles.py's
load_model() checks whether a checkpoint holds a pickled nn.Module, and
falls back to `--model-factory module:function` when it doesn't. Every
checkpoint this project actually writes (transformer.py's best_model.pt /
final_model.pt, extract_from_resume.py's extracted_model.pt) is a plain
state_dict, so the factory path is always required -- but no importable
builder function existed anywhere in the repo before this file, and
transformer.py itself cannot serve as one: it calls argparse.parse_args()
and starts loading data / training at module scope with no
`if __name__ == "__main__":` guard, so `import transformer` runs the whole
training script against the attention analysis's command-line arguments.

SpatialEmbedding and VanillaTransformer below are copied verbatim from
transformer.py so the reconstructed architecture is bit-identical -- the
same convention compute_random_baselines.py and analysis/metrics.py already
use ("Model code ... copied verbatim from metrics.py to guarantee identical
behavior. If you change metrics.py, mirror the change here."). If you
change the model in transformer.py, mirror the change here too.

Usage
-----
    python extract_attention_profiles.py --runs "final_checkpoints/**/best_model.pt" \\
        --data-root data --out profiles/trained \\
        --model-factory model_adapter:build_model_from_checkpoint \\
        --attention-extractor model_adapter:extract_attention

Run from analysis/what_do_heads_learn (matching info.md's run order) so plain module-name
import resolves; load_model() adds os.getcwd() to sys.path itself.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Verbatim from transformer.py (SpatialEmbedding, VanillaTransformer).
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# The actual adapter.
# --------------------------------------------------------------------------


def build_model_from_checkpoint(ckpt: dict) -> nn.Module:
    """Reconstruct a VanillaTransformer from a transformer.py-style checkpoint.

    Expects ckpt to hold "model_state_dict" (as written by transformer.py's
    best_model.pt / final_model.pt / resume_state.pt, or by
    extract_from_resume.py's extracted_model.pt) and "args" (vars(argparse
    Namespace) from that training run, for n_layers/n_heads/d_model/d_ff).

    NX is read from the state_dict's positional-encoding buffer
    (embedding.pe has shape (1, NX, d_model)) rather than hardcoded, so this
    keeps working if NX ever changes from the 64 used throughout the rest of
    the project (metrics.py, compute_random_baselines.py both hardcode 64).
    """
    run_args = ckpt["args"]
    state = ckpt["model_state_dict"]
    nx = state["embedding.pe"].shape[1]

    model = VanillaTransformer(
        nx=nx,
        n_layers=run_args["n_layers"],
        n_heads=run_args["n_heads"],
        d_model=run_args["d_model"],
        d_ff=run_args["d_ff"],
        dropout=run_args.get("dropout", 0.1),
    )
    model.load_state_dict(state)
    return model


def extract_attention(model: "VanillaTransformer", batch) -> np.ndarray:
    """Return attention of shape (n_layers, n_heads, N, N), averaged over batch.

    attention_common.extract_attention (the default attention-capture routine)
    patches nn.MultiheadAttention.forward and F.scaled_dot_product_attention
    -- but confirmed against a real checkpoint, neither ever fires here:
    nn.TransformerEncoderLayer takes an internal fast path in eval mode that
    calls neither of those from Python, so attention_common's patches capture
    nothing and extract_attention() raises "no attention captured". This is
    exactly the case info.md's "Two seams" section warns about, and exactly
    the problem metrics.py's collect_attn() and
    compute_random_baselines.py's collect_attn_mean() already solved there:
    a forward hook on each encoder layer that re-runs self_attn directly
    with need_weights=True, bypassing the fast path instead of trying to
    intercept it.

    Pass this as extract_attention_profiles.py's --attention-extractor
    model_adapter:extract_attention.
    """
    import torch

    n_layers = len(model.transformer.layers)
    buffers = [None] * n_layers

    def make_hook(idx):
        def hook(module, inp, out):
            with torch.no_grad():
                # norm_first=True: self_attn runs on norm1(src) during the
                # real forward pass, not on the layer's raw input.
                normed = module.norm1(inp[0])
                _, w = module.self_attn(normed, normed, normed,
                                         need_weights=True,
                                         average_attn_weights=False)
                buffers[idx] = w.mean(dim=0).detach().cpu().numpy()  # (H, N, N)
        return hook

    hooks = [layer.register_forward_hook(make_hook(i))
             for i, layer in enumerate(model.transformer.layers)]
    try:
        model.eval()
        with torch.no_grad():
            model(batch)
    finally:
        for h in hooks:
            h.remove()

    if any(b is None for b in buffers):
        raise RuntimeError(
            "no attention captured for one or more layers; model does not "
            "match the VanillaTransformer structure this hook assumes "
            "(model.transformer.layers[i].{norm1,self_attn})"
        )
    return np.stack(buffers, axis=0)  # (n_layers, n_heads, N, N)
