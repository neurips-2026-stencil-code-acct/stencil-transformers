"""Synthetic recovery test for the mechanistic interventions.

The failure mode this analysis has to rule out is silent no-ops. A substitution that never
reached the model and a substitution that genuinely does not change the loss
produce identical numbers, and only one of them is a result. Four checks:

  fidelity      the reimplemented MultiheadAttention forward reproduces
                PyTorch's own output to numerical precision when no
                intervention is requested. Everything else is meaningless if
                this fails, because the "original" baseline would not be the
                model's actual behaviour.
  recovery      the Jacobian of an exactly linear model equals its weight
                matrix, with zero circulant residual and zero stencil error.
  sensitivity   substituting attention in a model that uses attention changes
                the output.
  specificity   substituting attention in a model that computes attention but
                discards it changes nothing, while still reporting that
                interception occurred.

    python selftest.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "what_do_heads_learn"
    ),
)
import attention_common as C
from mechanistic_model import (AttentionControl, LayerAblation, model_output, to_batch,
                      head_output_means, circulant_from_profile)
from jacobian_analysis import jacobian_at, analyse_block

N, D_MODEL, N_HEAD = 16, 8, 2
torch.manual_seed(0)


class LinearOp(nn.Module):
    """Exactly the finite-difference operator, as a matrix multiply."""

    def __init__(self, weights, offsets):
        super().__init__()
        row = np.zeros(N)
        row[np.asarray(offsets) % N] = weights
        w = np.stack([np.roll(row, i) for i in range(N)])
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32))

    def forward(self, x):
        return x @ self.w.T


class TinyAttn(nn.Module):
    """Minimal attention model; `use_attn=False` computes it and throws it away."""

    def __init__(self, use_attn=True):
        super().__init__()
        self.embed = nn.Linear(1, D_MODEL)
        # Positional embeddings are not decoration here. Without them every
        # token differs only along the single direction embed maps into, the
        # keys are nearly collinear, and the model's own attention is already
        # close to uniform. Substituting uniform attention would then change
        # almost nothing and the sensitivity check would pass vacuously.
        self.pos = nn.Parameter(torch.randn(1, N, D_MODEL))
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEAD, batch_first=True)
        self.out = nn.Linear(D_MODEL, 1)
        self.use_attn = use_attn

    def forward(self, x):
        h = self.embed(x.unsqueeze(-1)) + self.pos
        a, _ = self.attn(h, h, h)
        return self.out(a if self.use_attn else h).squeeze(-1)


class EncoderModel(nn.Module):
    """nn.TransformerEncoder, which takes a fused fast path in eval mode."""

    def __init__(self, n_layers=2):
        super().__init__()
        self.emb = nn.Linear(1, D_MODEL)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D_MODEL, N_HEAD, 32, batch_first=True,
                                       dropout=0.0), n_layers)
        self.dec = nn.Linear(D_MODEL, 1)

    def forward(self, x):
        return self.dec(self.enc(self.emb(x.unsqueeze(-1)))).squeeze(-1)


def check_fidelity():
    x = torch.randn(32, N)
    for name, model, expect_calls in [("TinyAttn", TinyAttn().eval(), 1),
                                      ("TransformerEncoder",
                                       EncoderModel().eval(), 2)]:
        with torch.no_grad():
            ref = model(x)
            ctl = AttentionControl()
            with ctl:
                got = model(x)
            ctl.assert_intercepted()
        err = float((ref - got).abs().max())
        rel = err / float(ref.abs().max())
        print(f"fidelity     {name:20s} max |patched - pytorch| = {err:.3e} "
              f"(rel {rel:.2e}, {ctl.n_calls} interceptions)")
        assert rel < 1e-5, f"{name}: patched forward diverges, rel {rel}"
        assert ctl.n_calls == expect_calls, \
            f"{name}: expected {expect_calls} interceptions, got {ctl.n_calls}"


def check_recovery():
    print("\nrecovery     Jacobian of an exactly linear model")
    offsets = C.offsets_full(N)
    for name, w in [("heat r=0.25", np.array([0.25, 0.5, 0.25])),
                    ("lf nu=0.25", np.array([0.625, 0.0, 0.375]))]:
        model = LinearOp(w, [-1, 0, 1]).eval()
        x0 = np.random.default_rng(0).normal(size=(1, N)).astype(np.float32)
        j = jacobian_at(model, x0[0], "cpu")[:, 0, :]
        stats, prof = analyse_block(j, w, np.array([-1, 0, 1]), offsets)
        print(f"  {name:12s} circ_resid={stats['circulant_residual']:.2e}  "
              f"rel_err={stats['stencil_rel_error']:.2e}  "
              f"row_sum={stats['row_sum']:.4f}  "
              f"coefs=({stats['coef_-1']:.4f}, {stats['coef_+0']:.4f}, "
              f"{stats['coef_+1']:.4f})")
        assert stats["circulant_residual"] < 1e-5
        assert stats["stencil_rel_error"] < 1e-5
        assert stats["locality"] > 0.999


def check_sensitivity_and_specificity():
    x = torch.randn(64, N)
    subs = {
        "uniform": np.full((N, N), 1.0 / N),
        "stencil": circulant_from_profile(np.array([0.25, 0.5, 0.25]),
                                          [-1, 0, 1], N),
        "identity": np.eye(N),
    }
    print("\nsensitivity  model that uses attention")
    model = TinyAttn(use_attn=True).eval()
    with torch.no_grad():
        base = model(x)
    scale = float(base.abs().mean())
    deltas = {}
    for name, m in subs.items():
        ctl = AttentionControl(substitute=m)
        with torch.no_grad(), ctl:
            got = model(x)
        ctl.assert_intercepted()
        deltas[name] = float((got - base).abs().mean()) / scale
        print(f"  {name:9s} relative change = {deltas[name]:.4f}")
    assert min(deltas.values()) > 1e-3, "substitution had no effect on a model "\
                                        "that uses attention"

    print("\nspecificity  model that computes attention but discards it")
    model = TinyAttn(use_attn=False).eval()
    with torch.no_grad():
        base = model(x)
    for name, m in subs.items():
        ctl = AttentionControl(substitute=m)
        with torch.no_grad(), ctl:
            got = model(x)
        ctl.assert_intercepted()
        d = float((got - base).abs().max()) / max(float(base.abs().mean()), 1e-12)
        print(f"  {name:9s} relative max change = {d:.3e} "
              f"({ctl.n_calls} interceptions)")
        assert d < 1e-6, "substitution changed a model that ignores attention"
    print("  Interception still occurred, so a null result here is a finding "
          "rather than a plumbing failure. That is the distinction the "
          "assert_intercepted check exists to preserve.")


def check_head_and_layer_ablation():
    print("\nablation")
    model = TinyAttn(use_attn=True).eval()
    x = np.random.default_rng(1).normal(size=(64, 1, N)).astype(np.float32)
    means = head_output_means(model, x, "cpu")
    assert len(means) == N_HEAD, f"expected {N_HEAD} head means, got {len(means)}"

    with torch.no_grad():
        base = model_output(model, to_batch(x, "cpu"))
    for h in range(N_HEAD):
        ctl = AttentionControl(ablate_heads={0: [h]}, head_means=means)
        with torch.no_grad(), ctl:
            got = model_output(model, to_batch(x, "cpu"))
        d = float((got - base).abs().mean())
        print(f"  head {h} mean-ablated: mean |change| = {d:.5f}")
        assert d > 1e-5

    enc = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(D_MODEL, N_HEAD, 16, batch_first=True,
                                   dropout=0.0), 2).eval()
    h = torch.randn(8, N, D_MODEL)
    with torch.no_grad():
        full = enc(h)
        with LayerAblation(enc, [0, 1]):
            ablated = enc(h)
    d = float((ablated - h).abs().max())
    print(f"  both encoder layers ablated: max |output - input| = {d:.3e}, "
          f"vs |full - input| = {float((full - h).abs().max()):.4f}")
    assert d < 1e-6, "identity replacement did not pass the input through"


def main():
    check_fidelity()
    check_recovery()
    check_sensitivity_and_specificity()
    check_head_and_layer_ablation()
    print("\nself-test passed")


if __name__ == "__main__":
    main()
