"""Reviewed shared machinery for interventions on a trained model.

Earlier analyses read saved attention patterns. This analysis runs the model, so it needs to reach inside
the attention computation. Two interception paths are installed, mirroring
attention_common.AttentionCapture:

  * nn.MultiheadAttention, whose forward is replaced by an explicit
    reimplementation from in_proj / out_proj. Wrapping is not enough here:
    substitution and head ablation have to act between the softmax and the
    value multiply, which the stock forward does not expose.
  * F.scaled_dot_product_attention, for custom modules that route through it.

If a model computes softmax explicitly inside its own attention module,
neither fires, and `assert_intercepted` raises rather than silently reporting
that an intervention had no effect. That distinction matters more here than
anywhere else in this codebase: a substitution that never happened and a
substitution that genuinely does not change the loss produce identical numbers,
and the second is a publishable claim while the first is a bug.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# The attention-analysis modules are the single source of truth; import them from
# ../what_do_heads_learn rather than keeping a duplicate copy here (see
# analysis/parameter_sensitive/README.md's
# README note on cross-imports for why a duplicate is the wrong fix).
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "what_do_heads_learn"
    ),
)
import attention_common as C


# --------------------------------------------------------------------------
# model and data loading (same seams as extract_attention_profiles)
# --------------------------------------------------------------------------

def load_model(ckpt_path: str, device: str, factory: str | None = None):
    import torch

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if factory:
        mod_name, fn_name = factory.split(":")
        sys.path.insert(0, os.getcwd())
        model = getattr(__import__(mod_name, fromlist=[fn_name]), fn_name)(ckpt)
    elif isinstance(ckpt, torch.nn.Module):
        model = ckpt
    elif isinstance(ckpt, dict) and isinstance(ckpt.get("model"), torch.nn.Module):
        model = ckpt["model"]
    else:
        raise RuntimeError(
            f"{ckpt_path} holds a state_dict; pass --model-factory module:function"
        )
    return model.to(device).eval()


def load_eval_pairs(data_dir: str, n_pairs: int, n_input_steps: int = 1,
                    seed: int = 0):
    """Held-out (input, target) pairs. Returns (x, y) with x shaped

        (n_pairs, n_input_steps, N)   and   y shaped (n_pairs, N).

    Needs the time axis intact, i.e. (n_traj, n_t, N).
    """
    for name in ["test.npy", "test_u.npy", "u_test.npy", "trajectories.npy",
                 "u.npy", "train.npy"]:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        arr = np.asarray(np.load(path, mmap_mode="r"))
        if arr.ndim < 3:
            raise ValueError(
                f"{path} has shape {arr.shape}; this analysis needs (n_traj, n_t, N) so "
                "inputs can be paired with the next step")
        k = n_input_steps
        xs, ys = [], []
        for t in range(k, arr.shape[1]):
            xs.append(arr[:, t - k:t])
            ys.append(arr[:, t])
        x = np.concatenate(xs, 0).astype(np.float32)
        y = np.concatenate(ys, 0).astype(np.float32)
        if len(x) > n_pairs:
            idx = np.sort(np.random.default_rng(seed).choice(len(x), n_pairs, False))
            x, y = x[idx], y[idx]
        return x, y
    raise FileNotFoundError(f"no trajectory array in {data_dir}")


def to_batch(x: np.ndarray, device: str, squeeze_single_step: bool = True):
    """Shape inputs for the model's forward.

    Default drops a singleton time axis, so a one-step model receives (B, N).
    Adjust here if the forward expects a channel axis instead.
    """
    import torch

    t = torch.from_numpy(np.asarray(x, np.float32)).to(device)
    if squeeze_single_step and t.dim() == 3 and t.shape[1] == 1:
        t = t[:, 0]
    return t


def model_output(model, batch):
    """Model prediction as (B, N), tolerating a trailing singleton axis."""
    out = model(batch)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.dim() == 3 and out.shape[-1] == 1:
        out = out[..., 0]          # (B, N, 1) -> (B, N)
    elif out.dim() == 3:
        out = out[:, -1]           # (B, T, N) -> last predicted step
    return out


# --------------------------------------------------------------------------
# attention interception
# --------------------------------------------------------------------------

class AttentionControl:
    """Intercept attention to capture it, substitute it, or ablate heads.

    Parameters
    ----------
    substitute : array (N, N) or (n_layers, N, N) or None
        Row-stochastic matrix to use in place of the softmax output. A single
        matrix is broadcast to every intercepted layer.
    layers : set of int or None
        Restrict the intervention to these layer indices (order of execution).
    ablate_heads : dict {layer_index: [head, ...]}
        Replace those heads' per-head outputs with `head_means`.
    head_means : dict {(layer, head): array (d_head,)}
        Dataset-mean output for each ablated head. Mean ablation, not zeroing:
        zeroing removes the head's mean contribution as well and conflates
        "this head carries no information" with "this head carries a constant
        the rest of the network depends on".
    """

    def __init__(self, substitute=None, layers=None, ablate_heads=None,
                 head_means=None, capture=False, capture_head_outputs=False):
        self.substitute = substitute
        self.layers = layers
        self.ablate_heads = ablate_heads or {}
        self.head_means = head_means or {}
        self.capture = capture
        self.capture_head_outputs = capture_head_outputs
        self.attentions = []
        self.head_outputs = []
        self.n_calls = 0
        self._patches = []

    # -- helpers ----------------------------------------------------------

    def _sub_for(self, idx, n, device, dtype):
        import torch

        if self.substitute is None:
            return None
        if self.layers is not None and idx not in self.layers:
            return None
        s = np.asarray(self.substitute)
        m = s if s.ndim == 2 else s[idx]
        if m.shape[-1] != n:
            raise ValueError(
                f"substitute matrix is {m.shape[-1]}x{m.shape[-1]} but the "
                f"sequence length is {n}")
        return torch.as_tensor(m, device=device, dtype=dtype)

    def _apply(self, idx, attn, value):
        """attn: (B, H, N, N); value: (B, H, N, d). Returns per-head output."""
        import torch

        n = attn.shape[-1]
        sub = self._sub_for(idx, n, attn.device, attn.dtype)
        if sub is not None:
            attn = sub.expand(attn.shape[0], attn.shape[1], n, n)
        if self.capture:
            self.attentions.append(attn.detach().mean(0).cpu().numpy())
        out = attn @ value
        for h in self.ablate_heads.get(idx, []):
            mean = self.head_means.get((idx, h))
            if mean is None:
                raise KeyError(f"no head mean recorded for layer {idx} head {h}")
            out[:, h] = torch.as_tensor(mean, device=out.device, dtype=out.dtype)
        if self.capture_head_outputs:
            self.head_outputs.append(out.detach().mean(dim=(0, 2)).cpu().numpy())
        return out

    # -- context ----------------------------------------------------------

    def __enter__(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        ctl = self
        self._patches.extend(C.force_slow_path_patches())

        orig_mha = nn.MultiheadAttention.forward

        def mha_forward(mod, query, key=None, value=None, **kw):
            # This explicit implementation is deliberately limited to the
            # unmasked self-attention used by this project's
            # VanillaTransformer. Computing K/V from query is not a valid
            # cross-attention replacement, and ignoring masks changes the
            # model being evaluated, so refuse both cases.
            key = query if key is None else key
            value = key if value is None else value
            if key is not query or value is not query:
                raise NotImplementedError(
                    "AttentionControl currently supports only unmasked "
                    "self-attention; cross-attention needs separate Q/K/V handling"
                )
            if (kw.get("attn_mask") is not None or
                    kw.get("key_padding_mask") is not None or
                    bool(kw.get("is_causal", False))):
                raise NotImplementedError(
                    "AttentionControl does not implement masks or causal "
                    "attention; refusing a non-faithful intervention"
                )
            if getattr(mod, "bias_k", None) is not None or \
                    getattr(mod, "add_zero_attn", False):
                raise NotImplementedError(
                    "bias_k / add_zero_attn are not supported by the "
                    "reimplementation; the intervention would silently differ "
                    "from the model's own forward")
            x = query
            if not mod.batch_first:
                x = x.transpose(0, 1)
            b, n, e = x.shape
            h = mod.num_heads
            d = e // h

            if getattr(mod, "_qkv_same_embed_dim", True):
                qkv = F.linear(x, mod.in_proj_weight, mod.in_proj_bias)
                q, k, v = qkv.chunk(3, dim=-1)
            else:
                bq, bk, bv = (mod.in_proj_bias.chunk(3)
                              if mod.in_proj_bias is not None else (None,) * 3)
                q = F.linear(x, mod.q_proj_weight, bq)
                k = F.linear(x, mod.k_proj_weight, bk)
                v = F.linear(x, mod.v_proj_weight, bv)

            def split(t):
                return t.view(b, n, h, d).transpose(1, 2)

            q, k, v = split(q), split(k), split(v)
            attn = torch.softmax((q @ k.transpose(-2, -1)) / (d ** 0.5), dim=-1)

            idx = ctl.n_calls
            ctl.n_calls += 1
            out = ctl._apply(idx, attn, v)

            out = out.transpose(1, 2).reshape(b, n, e)
            out = mod.out_proj(out)
            if not mod.batch_first:
                out = out.transpose(0, 1)
            return out, None

        nn.MultiheadAttention.forward = mha_forward
        self._patches.append((nn.MultiheadAttention, "forward", orig_mha))

        orig_sdpa = F.scaled_dot_product_attention

        def sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, **kw):
            if is_causal or dropout_p != 0.0:
                raise NotImplementedError(
                    "AttentionControl does not implement causal or dropout "
                    "scaled-dot-product attention"
                )
            s = scale if scale is not None else 1.0 / (query.shape[-1] ** 0.5)
            logits = (query @ key.transpose(-2, -1)) * s
            if attn_mask is not None:
                logits = logits + (
                    attn_mask if attn_mask.dtype != torch.bool
                    else torch.zeros_like(logits).masked_fill(
                        ~attn_mask, float("-inf")))
            attn = torch.softmax(logits, dim=-1)
            idx = ctl.n_calls
            ctl.n_calls += 1
            return ctl._apply(idx, attn, value)

        F.scaled_dot_product_attention = sdpa
        self._patches.append((F, "scaled_dot_product_attention", orig_sdpa))
        return self

    def __exit__(self, *exc):
        for obj, name, orig in self._patches:
            setattr(obj, name, orig)
        self._patches.clear()
        return False

    def assert_intercepted(self):
        """A no-op intervention and a genuine null result must not be allowed
        to produce identical numbers (Kriegeskorte et al., 2009's "double
        dipping" concern is the same class of problem in spirit: a validity
        check has to be structurally incapable of passing vacuously; see
        README References and the historical alignment analysis's use of the same
        citation for the alignment null).
        """
        if self.n_calls == 0:
            raise RuntimeError(
                "no attention was intercepted, so every intervention below "
                "is a no-op reported as a null result. Replace the attention "
                "path in mechanistic_model.AttentionControl with one matching your "
                "module, or route the model through nn.MultiheadAttention.")


class LayerAblation:
    """Replace whole sublayers with the identity.

    Registers forward hooks that return the module's input. Which modules count
    as 'layers' is matched by class name, defaulting to anything containing
    'TransformerEncoderLayer' or 'Block'.
    """

    def __init__(self, model, layer_indices, name_pattern="EncoderLayer|Block"):
        import re

        self.modules = [m for m in model.modules()
                        if re.search(name_pattern, type(m).__name__)]
        self.indices = set(layer_indices)
        self.handles = []

    def __enter__(self):
        def make_hook():
            def hook(mod, inputs, output):
                x = inputs[0]
                if isinstance(output, tuple):
                    return (x,) + output[1:]
                return x
            return hook

        for i, mod in enumerate(self.modules):
            if i in self.indices:
                self.handles.append(mod.register_forward_hook(make_hook()))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate(model, x, y, device, batch_size=256, control_kwargs=None,
             layer_ablation=None, layer_pattern="EncoderLayer|Block"):
    """Mean squared error over held-out pairs, under an optional intervention."""
    import torch

    total, count, ctl = 0.0, 0, None
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = to_batch(x[i:i + batch_size], device)
            yb = torch.from_numpy(y[i:i + batch_size]).to(device)
            if control_kwargs is not None:
                ctl = AttentionControl(**control_kwargs)
                with ctl:
                    if layer_ablation:
                        with LayerAblation(model, layer_ablation, layer_pattern):
                            pred = model_output(model, xb)
                    else:
                        pred = model_output(model, xb)
                ctl.assert_intercepted()
            elif layer_ablation:
                with LayerAblation(model, layer_ablation, layer_pattern):
                    pred = model_output(model, xb)
            else:
                pred = model_output(model, xb)
            total += float(((pred - yb) ** 2).sum())
            count += yb.numel()
    return total / max(count, 1)


def baseline_mse(x, y):
    """MSE of the persistence predictor u^{n+1} = u^n.

    Interventions have to be scored against something. Raw MSE is
    uninterpretable on its own, and persistence is the right floor here because
    all three schemes are close to the identity at small r, so a model can look
    accurate while having learned very little. `excess` (built from this and
    mse_original) is a mean-squared-error skill score (Murphy, 1988; see
    README References) against persistence rather than climatology.
    """
    return float(((x[:, -1] - y) ** 2).mean())


def skill_guard(mse_original, mse_persistence, label=""):
    """Warn when the excess-error normaliser is unstable.

    `excess` divides by (persistence - original), the model's skill margin. All
    three schemes are close to the identity at small r, so persistence is a
    strong baseline and that margin can be a thin difference of two similar
    numbers. When it is, excess is a ratio of noise and produces values in the
    hundreds that mean nothing. The raw ratio mse / mse_original stays
    interpretable and should be read instead.
    """
    skill = (mse_persistence - mse_original) / max(mse_persistence, 1e-30)
    if skill < 0.1:
        print(f"warning: {label}the model beats persistence by only "
              f"{skill:.1%}, so 'excess' divides by a near-zero margin and is "
              "unstable. Read the 'ratio' column instead.", file=sys.stderr)
    return skill


def head_output_means(model, x, device, batch_size=256):
    """Dataset-mean per-head attention output, for mean ablation.

    Mean ablation rather than zero ablation, matching the same distinction
    made in the mechanistic-interpretability circuit-analysis literature
    (Wang et al., 2023; see README References): zeroing conflates "this head
    carries no information" with "this head carries a constant the rest of
    the network depends on."
    """
    import torch

    sums, counts = {}, {}
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = to_batch(x[i:i + batch_size], device)
            ctl = AttentionControl(capture_head_outputs=True)
            with ctl:
                model_output(model, xb)
            ctl.assert_intercepted()
            for layer, ho in enumerate(ctl.head_outputs):
                sums[layer] = sums.get(layer, 0.0) + ho * len(xb)
                counts[layer] = counts.get(layer, 0) + len(xb)
    out = {}
    for layer, s in sums.items():
        mean = s / counts[layer]              # (n_heads, d_head)
        for h in range(mean.shape[0]):
            out[(layer, h)] = mean[h]
    return out


def circulant_from_profile(profile, offsets, n):
    """Row-stochastic circulant attention with the given offset profile."""
    row = np.zeros(n, dtype=np.float64)
    p = np.abs(np.asarray(profile, dtype=np.float64))
    row[np.asarray(offsets) % n] = p
    s = row.sum()
    row = row / s if s > 0 else np.full(n, 1.0 / n)
    return np.stack([np.roll(row, i) for i in range(n)], axis=0)
