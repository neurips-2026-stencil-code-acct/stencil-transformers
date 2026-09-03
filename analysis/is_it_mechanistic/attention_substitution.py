"""Replace the attention matrix and measure what the model loses.

Substitutions, all row-stochastic circulants except where noted:

    original    the model's own attention, as a control
    frozen      the model's own attention averaged over the test set, i.e. its
                own pattern stripped of input dependence
    stencil     the true update rule, normalised to sum to one per row
    acf         the empirical autocorrelation profile
    xcorr       the empirical input-output cross-correlation profile
    uniform     1/N everywhere
    random      a random row-stochastic circulant
    identity    attend only to self

The `frozen` control is the one that decides how to read everything else. If
freezing the model's own attention costs nothing, then attention is effectively
a fixed matrix in this model, no dynamic routing is happening, and the whole
substitution comparison is a question about which fixed matrix works. If
freezing is costly, attention is doing input-dependent work and a fixed
substitute cannot match it regardless of which profile it uses. Without this
control, a finding that every substitution degrades accuracy is ambiguous
between "attention carries the operator" and "attention carries input-dependent
information of some other kind".

Substitutions are applied to all layers by default and per layer with
--per-layer, since a matrix that is harmless in one layer may be fatal in
another.

Usage
-----
    python attention_substitution.py --runs "runs/**/best.pt" --data-root data \
        --predictors predictors.npz --xcorr xcorr.npz --out results
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
from mechanistic_model import (AttentionControl, load_model, load_eval_pairs, to_batch,
                      model_output, evaluate, baseline_mse, skill_guard,
                      circulant_from_profile)


def build_substitutes(key, n, pred, xcorr, rng):
    """Named row-stochastic circulants for one condition."""
    offsets = C.offsets_full(n)
    subs = {}

    if pred is not None and f"{key}/stencil_raw" in pred:
        w = np.asarray(pred[f"{key}/stencil_raw"])
        off = np.asarray(pred[f"{key}/stencil_offsets"])
        subs["stencil"] = circulant_from_profile(np.abs(w), off, n)
    if pred is not None and f"{key}/acf_ring" in pred:
        ring = np.abs(np.asarray(pred[f"{key}/acf_ring"]))
        subs["acf"] = circulant_from_profile(ring[offsets % n], offsets, n)
    if xcorr is not None and f"{key}/xcorr_ring" in xcorr:
        ring = np.abs(np.asarray(xcorr[f"{key}/xcorr_ring"]))
        subs["xcorr"] = circulant_from_profile(ring[offsets % n], offsets, n)

    subs["uniform"] = np.full((n, n), 1.0 / n)
    r = rng.random(n)
    subs["random"] = circulant_from_profile(r, offsets, n)
    subs["identity"] = np.eye(n)
    return subs


def mean_attention(model, x, device, batch_size=256):
    """Per-layer attention averaged over heads and the test set."""
    import torch

    sums, count = {}, 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = to_batch(x[i:i + batch_size], device)
            ctl = AttentionControl(capture=True)
            with ctl:
                model_output(model, xb)
            ctl.assert_intercepted()
            for layer, a in enumerate(ctl.attentions):
                sums[layer] = sums.get(layer, 0.0) + a.mean(0) * len(xb)
            count += len(xb)
    return np.stack([sums[l] / count for l in sorted(sums)])


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
    ap.add_argument("--xcorr", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-pairs", type=int, default=2048)
    ap.add_argument("--n-input-steps", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--per-layer", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-factory", default=None)
    ap.add_argument("--run-regex", default=C.DEFAULT_RUN_REGEX)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.runs, recursive=True))
    if not paths:
        sys.exit(f"no checkpoints matched {args.runs}")
    pred = np.load(args.predictors) if os.path.exists(args.predictors) else None
    xcorr = np.load(args.xcorr) if args.xcorr and os.path.exists(args.xcorr) else None
    rng = np.random.default_rng(args.seed)

    rows = []
    for path in paths:
        meta = C.parse_run(path, args.run_regex)
        if meta is None:
            continue
        key = f"{meta.pde}_r{meta.r:g}"
        ddir = condition_data_dir(args.data_root, meta, args.data_template,
                                  args.condition_regex)
        x, y = load_eval_pairs(ddir, args.n_pairs, args.n_input_steps,
                               seed=meta.seed)
        model = load_model(path, args.device, args.model_factory)
        n = x.shape[-1]

        mse_orig = evaluate(model, x, y, args.device, args.batch_size)
        mse_persist = baseline_mse(x, y)
        skill = skill_guard(mse_orig, mse_persist,
                            f"{meta.pde} r={meta.r:g} seed {meta.seed}: ")

        subs = build_substitutes(key, n, pred, xcorr, rng)
        subs["frozen"] = mean_attention(model, x, args.device, args.batch_size)

        n_layers = len(subs["frozen"])
        targets = [(None, "all")] + ([(l, f"layer{l}") for l in range(n_layers)]
                                     if args.per_layer else [])

        for layers, tag in targets:
            sel = None if layers is None else {layers}
            for name, m in subs.items():
                mse = evaluate(model, x, y, args.device, args.batch_size,
                               control_kwargs={"substitute": m, "layers": sel})
                rows.append({
                    "pde": meta.pde, "r": meta.r, "seed": meta.seed,
                    "target": tag, "substitution": name, "mse": mse,
                    "mse_original": mse_orig, "mse_persistence": mse_persist,
                    "excess": (mse - mse_orig) / max(mse_persist - mse_orig, 1e-12),
                    "ratio": mse / max(mse_orig, 1e-30), "skill": skill,
                })
            rows.append({
                "pde": meta.pde, "r": meta.r, "seed": meta.seed, "target": tag,
                "substitution": "original", "mse": mse_orig,
                "mse_original": mse_orig, "mse_persistence": mse_persist,
                "excess": 0.0, "ratio": 1.0, "skill": skill,
            })
        del model

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "substitution.csv"), index=False)

    piv = (df[df.target == "all"]
           .pivot_table(index=["pde", "r"], columns="substitution",
                        values="excess", aggfunc="mean"))
    piv_ratio = (df[df.target == "all"]
                 .pivot_table(index=["pde", "r"], columns="substitution",
                              values="ratio", aggfunc="mean"))
    order = [c for c in ["original", "frozen", "stencil", "acf", "xcorr",
                         "uniform", "random", "identity"] if c in piv.columns]
    print("excess error, 0 = original attention, 1 = persistence baseline")
    print(piv[order].round(3).to_string())
    print("\nmse relative to original attention")
    print(piv_ratio[order].round(3).to_string())

    if "frozen" in piv.columns:
        fr = piv["frozen"].mean()
        print(f"\nfreezing the model's own attention costs {fr:.3f} of the "
              "original-to-persistence gap.")
        if fr < 0.05:
            print("Attention is effectively static in these models: input "
                  "dependence contributes almost nothing, so the comparison "
                  "below is about which fixed matrix works, not about routing.")
        else:
            print("Attention is doing input-dependent work, so no fixed "
                  "substitute can match the original and the fixed "
                  "substitutions should be compared against 'frozen', not "
                  "against 'original'.")
    print(f"\nwrote {args.out}/substitution.csv")


if __name__ == "__main__":
    main()
