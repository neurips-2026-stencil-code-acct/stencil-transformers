"""Shared conventions for the head-interpretation analysis.

Everything downstream operates on *relative-offset profiles*: probability
distributions over spatial offset d, obtained by averaging the attention
matrix along its diagonals (valid because the PDE setup is periodic and the
true operator is circulant). The conventions fixed here are the single
source of truth:

  1. attention profile   p[d] = mean_i A[i, (i+d) mod N]
  2. predictor profiles  built from |w|, renormalised over the same window
  3. divergences         computed on windowed, renormalised profiles
  4. gating              measured on the FULL profile, before windowing

Wave caveat: `stencil.npy` for the wave arm stores only the spatial part of
the leapfrog update, [r, 2-2r, r]; the u^{n-1} coefficient is not stored.
The attention analysis compares spatial attention within a timestep uniformly across all three
PDEs, so this is consistent, but it needs the footnote in the paper.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict

import numpy as np

# --------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------


def offsets_full(n: int) -> np.ndarray:
    """Signed offsets covering the whole ring, ordered ascending."""
    return np.arange(-(n // 2), n - n // 2)


def relative_profile(a: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Diagonal-average an attention matrix into an offset profile.

    Parameters
    ----------
    a : array, shape (..., N, N)
        Row-stochastic attention. Leading axes (layer, head, ...) are kept.
    offsets : array of int
        Signed offsets to evaluate.

    Returns
    -------
    array, shape (..., len(offsets))
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[-1]
    rows = np.arange(n)
    cols = (rows[:, None] + np.asarray(offsets)[None, :]) % n
    gathered = a[..., rows[:, None], cols]  # (..., N, len(offsets))
    return gathered.mean(axis=-2)


def restrict(p: np.ndarray, offsets: np.ndarray, half_width: int):
    """Window a profile to |d| <= half_width and renormalise.

    Returns (windowed_profile, kept_offsets, mass_inside_window). The retained
    mass is a diagnostic: a head whose attention lives mostly outside the
    window is being judged on a small slice of itself, and that should be
    visible in the results table rather than silently absorbed.
    """
    offsets = np.asarray(offsets)
    mask = np.abs(offsets) <= half_width
    q = np.asarray(p, dtype=np.float64)[..., mask]
    mass = q.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = q / np.where(mass[..., None] > 0, mass[..., None], 1.0)
    return q, offsets[mask], mass


def normalise_abs(w: np.ndarray) -> np.ndarray:
    """|w| renormalised to sum to one; the predictor convention."""
    w = np.abs(np.asarray(w, dtype=np.float64))
    s = w.sum(axis=-1, keepdims=True)
    return w / np.where(s > 0, s, 1.0)


# --------------------------------------------------------------------------
# divergences
# --------------------------------------------------------------------------


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-3) -> float:
    """KL(p || q) with q smoothed toward uniform.

    Smoothing is required because the Lax-Friedrichs stencil predictor has an
    exact zero at the centre, which is the whole point of that arm; without
    smoothing any centre-attending head scores infinity and the comparison
    degenerates.

    Kullback & Leibler, 1951, "On information and sufficiency",
    Annals of Mathematical Statistics 22(1). See info.md References.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    k = q.shape[-1]
    q = (1.0 - eps) * q + eps / k
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats. Bounded, symmetric, zero-safe.

    Lin, 1991, "Divergence measures based on the Shannon entropy", IEEE
    Trans. Information Theory 37(1). See info.md References.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def l2_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(p) - np.asarray(q)))


DIVERGENCES = {"js": js_divergence, "kl": kl_divergence, "l2": l2_distance}


def normalised_entropy(p: np.ndarray) -> np.ndarray:
    """H(p) / log(len(p)); 0 = delta, 1 = uniform. Computed on FULL profiles."""
    p = np.asarray(p, dtype=np.float64)
    k = p.shape[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log(p), 0.0)
    return terms.sum(axis=-1) / np.log(k)


# --------------------------------------------------------------------------
# predictor construction
# --------------------------------------------------------------------------


def stencil_row(weights: np.ndarray, stencil_offsets, n: int) -> np.ndarray:
    """Embed a short stencil into a length-N circulant row indexed by offset."""
    row = np.zeros(n, dtype=np.float64)
    row[np.asarray(stencil_offsets) % n] = np.asarray(weights, dtype=np.float64)
    return row


def spectral_row(weights: np.ndarray, stencil_offsets, n: int, kmax: int) -> np.ndarray:
    """Row of the true operator with its symbol truncated to |k| <= kmax.

    This is the operational definition of the 'spectral computation'
    hypothesis: the head implements the same operator, but band-limited, which
    produces a wide sinc-like profile with ringing rather than a compact
    3-point one. kmax -> N/2 recovers the stencil exactly, so the family nests
    the stencil hypothesis; the assignment step handles that with an explicit
    margin and a capped kmax grid.

    The truncated symbol is the discrete analogue of the Fourier-based
    stability/dissipation analysis LeVeque (2007) applies to these same
    finite-difference operators elsewhere in the project (see
    generate_heat_equation.py, generate_lax_friedrichs.py). See info.md
    References.
    """
    row = stencil_row(weights, stencil_offsets, n)
    symbol = np.fft.fft(row)
    freqs = np.fft.fftfreq(n, d=1.0 / n)
    symbol = np.where(np.abs(freqs) <= kmax, symbol, 0.0)
    return np.real(np.fft.ifft(symbol))


def spatial_acf(u: np.ndarray) -> np.ndarray:
    """Mean circular spatial autocorrelation over snapshots.

    u : array, shape (..., N). Each snapshot is de-meaned and normalised by its
    own lag-0 value before averaging, so high-amplitude early snapshots do not
    dominate the mean. Returns array of length N indexed by offset mod N.
    """
    x = np.asarray(u, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    n = x.shape[-1]
    f = np.fft.rfft(x, axis=-1)
    ac = np.fft.irfft((f * np.conj(f)).real, n=n, axis=-1)
    lag0 = ac[..., :1]
    good = np.squeeze(lag0, axis=-1) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        ac = ac / np.where(lag0 > 0, lag0, 1.0)
    ac = ac.reshape(-1, n)[good.reshape(-1)]
    if ac.size == 0:
        raise ValueError("all snapshots were constant; cannot form an ACF")
    return ac.mean(axis=0)


# --------------------------------------------------------------------------
# run discovery
# --------------------------------------------------------------------------

DEFAULT_RUN_REGEX = (
    r"(?P<pde>heat|wave|lf|advection|lax_friedrichs)"
    # (?:_new)? matches heat's on-disk "heat_new_<r>" convention (run_heat.py,
    # batch_metrics.py). The separator on both sides of (?:r|nu)? is required
    # because lf/wave's on-disk convention doubles it: "lax_friedrichs_r_<r>",
    # "wave_r_<r>" (run_friedrichs.py), not "lax_friedrichs_r<r>". Verified
    # against every directory actually on disk under data/ and
    # final_checkpoints/, including confirming heat_nlayers_<n> (a different,
    # unrelated sweep) still does not match.
    r"(?:_new)?[_-]?(?:r|nu)?[_-]?(?P<r>[0-9]*\.?[0-9]+)"
    r".*?seed[_-]?(?P<seed>[0-9]+)"
)

PDE_ALIASES = {
    "lf": "lf",
    "lax_friedrichs": "lf",
    "advection": "lf",
    "heat": "heat",
    "wave": "wave",
}


@dataclass
class RunMeta:
    pde: str
    r: float
    seed: int
    path: str

    def condition(self) -> str:
        return f"{self.pde}_r{self.r:g}"

    def to_dict(self):
        return asdict(self)


def parse_run(path: str, regex: str = DEFAULT_RUN_REGEX) -> RunMeta | None:
    """Pull (pde, r, seed) out of a checkpoint path. Returns None on no match."""
    m = re.search(regex, path.replace("\\", "/"), flags=re.IGNORECASE)
    if not m:
        return None
    pde = PDE_ALIASES.get(m.group("pde").lower(), m.group("pde").lower())
    return RunMeta(pde=pde, r=float(m.group("r")), seed=int(m.group("seed")), path=path)


# --------------------------------------------------------------------------
# attention-extraction seam
# --------------------------------------------------------------------------


def force_slow_path_patches():
    """Patches that stop nn.TransformerEncoder taking its fused fast path.

    In eval mode, nn.TransformerEncoderLayer dispatches to a fused kernel that
    computes attention internally, bypassing both MultiheadAttention.forward
    and scaled_dot_product_attention. Nothing is then interceptable, and the
    symptom is not an error but an absence: capture returns nothing, and any
    intervention silently becomes a no-op. nn.TransformerEncoder can also
    convert inputs to nested tensors, which fails for the same reason.

    Both forwards are replaced by their explicit equivalents, which route
    through MultiheadAttention and so through the interception layer. The
    replacements are numerically identical to the originals; the mechanistic self-test
    asserts that against a real nn.TransformerEncoder (0.0 absolute error).

    Confirmed to be the actual cause, not a hypothetical one: this project's
    VanillaTransformer uses nn.TransformerEncoderLayer, and extract_attention()
    raised "no attention captured" against a real checkpoint until this was
    applied (see model_adapter.py's now-redundant custom extract_attention,
    kept as a second, independently-verified capture path).

    Returns a list of (object, attribute, original) for restoration.
    """
    import torch.nn as nn

    patches = []

    orig_layer = nn.TransformerEncoderLayer.forward

    def layer_forward(mod, src, src_mask=None, src_key_padding_mask=None,
                      is_causal=False):
        # Signature is detected by inspection, never by trial call. Invoking
        # _sa_block to probe it would run a real attention pass, inflating the
        # interception count and shifting every layer index by one, which would
        # silently apply per-layer substitutions to the wrong layer.
        import inspect

        x = src
        try:
            takes_causal = "is_causal" in inspect.signature(
                mod._sa_block).parameters
        except (TypeError, ValueError):
            takes_causal = False

        def sa(h):
            if takes_causal:
                return mod._sa_block(h, src_mask, src_key_padding_mask,
                                     is_causal=is_causal)
            return mod._sa_block(h, src_mask, src_key_padding_mask)

        if mod.norm_first:
            x = x + sa(mod.norm1(x))
            x = x + mod._ff_block(mod.norm2(x))
        else:
            x = mod.norm1(x + sa(x))
            x = mod.norm2(x + mod._ff_block(x))
        return x

    nn.TransformerEncoderLayer.forward = layer_forward
    patches.append((nn.TransformerEncoderLayer, "forward", orig_layer))

    orig_enc = nn.TransformerEncoder.forward

    def enc_forward(mod, src, mask=None, src_key_padding_mask=None,
                    is_causal=None):
        out = src
        for layer in mod.layers:
            out = layer(out, src_mask=mask,
                        src_key_padding_mask=src_key_padding_mask,
                        is_causal=bool(is_causal))
        if mod.norm is not None:
            out = mod.norm(out)
        return out

    nn.TransformerEncoder.forward = enc_forward
    patches.append((nn.TransformerEncoder, "forward", orig_enc))
    return patches


class AttentionCapture:
    """Capture per-head attention from a vanilla PyTorch transformer.

    Three capture paths are installed:
      * force_slow_path_patches(), so nn.TransformerEncoderLayer/Encoder route
        through nn.MultiheadAttention instead of an uninterceptable fused
        kernel (see that function's docstring -- this is the fix for the
        actual failure mode hit against this project's real checkpoints);
      * nn.MultiheadAttention.forward, forced to need_weights=True and
        average_attn_weights=False;
      * F.scaled_dot_product_attention, recomputing softmax(QK^T/sqrt(d)).

    If your model computes softmax explicitly inside a custom module, none of
    these fire. In that case replace `extract_attention` below with the
    extraction routine already in metrics.py; it is the only function the
    rest of the attention analysis depends on, and it must return an array of shape
    (n_layers, n_heads, N, N) with rows summing to one.
    """

    def __init__(self):
        self.buffers = []
        self._patches = []

    def __enter__(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        buffers = self.buffers
        self._patches.extend(force_slow_path_patches())

        orig_mha = nn.MultiheadAttention.forward

        def mha_forward(mod, *args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            out = orig_mha(mod, *args, **kwargs)
            if isinstance(out, tuple) and out[1] is not None:
                w = out[1].detach()
                if w.dim() == 4:  # (B, H, N, N)
                    buffers.append(w.mean(dim=0).cpu().numpy())
            return out

        nn.MultiheadAttention.forward = mha_forward
        self._patches.append((nn.MultiheadAttention, "forward", orig_mha))

        orig_sdpa = F.scaled_dot_product_attention

        def sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, **kw):
            with torch.no_grad():
                s = scale if scale is not None else 1.0 / (query.shape[-1] ** 0.5)
                logits = (query @ key.transpose(-2, -1)) * s
                if attn_mask is not None:
                    logits = logits + (attn_mask if attn_mask.dtype != torch.bool
                                       else torch.zeros_like(logits).masked_fill(
                                           ~attn_mask, float("-inf")))
                w = torch.softmax(logits.float(), dim=-1).detach()
                if w.dim() == 4:
                    buffers.append(w.mean(dim=0).cpu().numpy())
            return orig_sdpa(query, key, value, attn_mask=attn_mask,
                             dropout_p=dropout_p, is_causal=is_causal, **kw)

        F.scaled_dot_product_attention = sdpa
        self._patches.append((F, "scaled_dot_product_attention", orig_sdpa))
        return self

    def __exit__(self, *exc):
        for obj, name, orig in self._patches:
            setattr(obj, name, orig)
        self._patches.clear()
        return False

    def stacked(self) -> np.ndarray:
        if not self.buffers:
            raise RuntimeError(
                "no attention captured; replace extract_attention() with the "
                "extraction routine from metrics.py"
            )
        return np.stack(self.buffers, axis=0)  # (n_layers, n_heads, N, N)


def extract_attention(model, batch) -> np.ndarray:
    """Return attention of shape (n_layers, n_heads, N, N), averaged over batch.

    Averaging over the batch before profiling is exact: the relative profile is
    linear in A, so mean-then-profile equals profile-then-mean.
    """
    import torch

    with AttentionCapture() as cap, torch.no_grad():
        model(batch)
        return cap.stacked()


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)
