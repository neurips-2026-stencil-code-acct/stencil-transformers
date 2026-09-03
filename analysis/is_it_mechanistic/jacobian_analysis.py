"""Extract the operator the model actually computes.

All three schemes are linear, so the trained model approximates a linear map
and that map can be read off directly as a Jacobian. This decides the paper's
central question without reference to attention at all, which is what makes it
worth running first: if the Jacobian recovers the stencil while the attention
profiles track autocorrelation, the result stops being "attention does not look
like the operator" and becomes "the operator is recovered, and it does not live
in the attention matrix".

Three things are measured, and all three are needed for that claim to hold.

  linearity      how well the local linearisation predicts held-out outputs.
                 A Jacobian is only the operator if the map is linear; softmax,
                 LayerNorm and the MLP are not, so this is checked rather than
                 assumed. Reported as R^2 of J(u - u0) + f(u0) against f(u),
                 and as the spread of J across base points.
  circulance     residual of J against its own diagonal average. The true
                 operator is translation invariant; a Jacobian that is not
                 circulant is not a finite-difference operator whatever its
                 rows look like.
  stencil match  the diagonal-averaged row against the true stencil, in
                 absolute terms rather than by correlation, since correlation
                 is shape-only and would score a scaled or near-uniform
                 operator as a perfect match.

Multi-step inputs are handled: the Jacobian is split per input timestep, so a
leapfrog model receiving (u^{n-1}, u^n) yields both blocks and the u^{n-1}
block can be checked against its predicted coefficient of -1.

Usage
-----
    python jacobian_analysis.py --runs "runs/**/best.pt" --data-root data \
        --predictors predictors.npz --out results
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "what_do_heads_learn"
    ),
)
import attention_common as C
from build_predictors import CONDITION_REGEX
from extract_attention_profiles import condition_data_dir
from mechanistic_model import load_model, load_eval_pairs, to_batch, model_output


def jacobian_at(model, x0, device):
    """d output / d input at one base point. Returns (N, n_steps, N)."""
    import torch

    xb = to_batch(x0[None], device).requires_grad_(True)

    def f(inp):
        return model_output(model, inp).reshape(-1)

    j = torch.autograd.functional.jacobian(f, xb, vectorize=True)
    return j.detach().cpu().numpy().reshape(j.shape[0], -1, x0.shape[-1])


def circulant_average(m: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Diagonal average of a square matrix, indexed by signed offset."""
    n = m.shape[-1]
    rows = np.arange(n)
    cols = (rows[:, None] + offsets[None, :]) % n
    return m[rows[:, None], cols].mean(axis=0)


def circulant_from_row(row: np.ndarray, offsets: np.ndarray, n: int) -> np.ndarray:
    full = np.zeros(n)
    full[np.asarray(offsets) % n] = row
    return np.stack([np.roll(full, i) for i in range(n)], axis=0)


def analyse_block(block: np.ndarray, true_stencil, stencil_offsets, offsets):
    """Statistics for one (output, input-timestep) Jacobian block."""
    n = block.shape[-1]
    prof = circulant_average(block, offsets)
    circ = circulant_from_row(prof, offsets, n)
    denom = np.linalg.norm(block)
    res = float(np.linalg.norm(block - circ) / denom) if denom > 0 else np.nan

    local = np.abs(prof[np.abs(offsets) <= 1]).sum()
    total = np.abs(prof).sum()
    out = {
        "circulant_residual": res,
        "locality": float(local / total) if total > 0 else np.nan,
        "row_sum": float(block.sum(axis=1).mean()),
        "row_sum_sd": float(block.sum(axis=1).std()),
    }
    for d in (-1, 0, 1):
        out[f"coef_{d:+d}"] = float(prof[offsets == d][0])

    if true_stencil is not None:
        target = np.zeros(len(offsets))
        for w, d in zip(true_stencil, stencil_offsets):
            target[offsets == d] = w
        err = np.linalg.norm(prof - target)
        scale = np.linalg.norm(target)
        out["stencil_rel_error"] = float(err / scale) if scale > 0 else np.nan
        out["stencil_corr"] = float(np.corrcoef(prof, target)[0, 1])
    return out, prof


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--data-template", default="{pde}_r{r}",
                    help="tried first; falls back to --condition-regex (see "
                         "extract_attention_profiles.condition_data_dir) when it "
                         "doesn't match anything on disk")
    ap.add_argument("--condition-regex", default=CONDITION_REGEX)
    ap.add_argument("--predictors", default="predictors.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-base-points", type=int, default=16)
    ap.add_argument("--n-linearity-probes", type=int, default=64)
    ap.add_argument("--n-input-steps", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-factory", default=None)
    ap.add_argument("--run-regex", default=C.DEFAULT_RUN_REGEX)
    args = ap.parse_args()

    import torch

    paths = sorted(glob.glob(args.runs, recursive=True))
    if not paths:
        sys.exit(f"no checkpoints matched {args.runs}")
    pred = np.load(args.predictors) if os.path.exists(args.predictors) else None

    rows, profiles = [], {}
    for path in paths:
        meta = C.parse_run(path, args.run_regex)
        if meta is None:
            continue
        key = f"{meta.pde}_r{meta.r:g}"
        ddir = condition_data_dir(args.data_root, meta, args.data_template,
                                  args.condition_regex)
        x, y = load_eval_pairs(ddir, args.n_linearity_probes,
                               args.n_input_steps, seed=meta.seed)
        model = load_model(path, args.device, args.model_factory)

        true_stencil = stencil_offsets = None
        if pred is not None and f"{key}/stencil_raw" in pred:
            true_stencil = np.asarray(pred[f"{key}/stencil_raw"])
            stencil_offsets = np.asarray(pred[f"{key}/stencil_offsets"])

        n = x.shape[-1]
        offsets = C.offsets_full(n)
        base_idx = np.random.default_rng(meta.seed).choice(
            len(x), min(args.n_base_points, len(x)), replace=False)

        jacs = np.stack([jacobian_at(model, x[i], args.device) for i in base_idx])
        jmean = jacs.mean(0)                       # (N, n_steps, N)
        spread = float(jacs.std(0).mean() / (np.abs(jmean).mean() + 1e-12))

        # linearity: does the mean Jacobian predict held-out outputs?
        with torch.no_grad():
            u0 = x[base_idx[0]]
            f0 = model_output(model, to_batch(u0[None], args.device)).cpu().numpy()[0]
            preds = model_output(model, to_batch(x, args.device)).cpu().numpy()
        flat = (x - u0[None]).reshape(len(x), -1)
        lin = f0[None] + flat @ jmean.reshape(jmean.shape[0], -1).T
        ss_res = float(((preds - lin) ** 2).sum())
        ss_tot = float(((preds - preds.mean(0)) ** 2).sum())
        r2_lin = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        for step in range(jmean.shape[1]):
            block = jmean[:, step, :]
            target = true_stencil if step == jmean.shape[1] - 1 else None
            stats, prof = analyse_block(block, target, stencil_offsets, offsets)
            stats.update({
                "pde": meta.pde, "r": meta.r, "seed": meta.seed,
                "input_step": step - (jmean.shape[1] - 1),
                "r2_linearity": r2_lin, "jacobian_spread": spread,
            })
            rows.append(stats)
            profiles[f"{key}_seed{meta.seed}_step{step}"] = prof

        del model

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "jacobian.csv"), index=False)
    np.savez_compressed(os.path.join(args.out, "jacobian_profiles.npz"),
                        offsets=C.offsets_full(n), **profiles)

    cur = df[df.input_step == 0]
    cols = ["pde", "r", "r2_linearity", "jacobian_spread", "circulant_residual",
            "locality", "row_sum", "coef_-1", "coef_+0", "coef_+1"]
    if "stencil_rel_error" in cur:
        cols += ["stencil_rel_error", "stencil_corr"]
    print(cur.groupby(["pde", "r"])[
        [c for c in cols if c not in ("pde", "r")]].mean().round(4).to_string())

    if len(cur):
        bad = cur[cur["r2_linearity"] < 0.95]
        if len(bad):
            print(f"\nwarning: linearisation R2 below 0.95 in "
                  f"{len(bad)}/{len(cur)} runs. The Jacobian is then a local "
                  "derivative, not the operator, and 'the model recovers the "
                  "stencil' does not follow from a matching row.", file=sys.stderr)
        far = cur[cur["circulant_residual"] > 0.2]
        if len(far):
            print(f"warning: Jacobian is not translation invariant in "
                  f"{len(far)}/{len(cur)} runs (circulant residual > 0.2); the "
                  "diagonal-averaged row is a summary, not the operator.",
                  file=sys.stderr)
    print(f"\nwrote {args.out}/jacobian.csv, jacobian_profiles.npz")


if __name__ == "__main__":
    main()
