"""Synthetic recovery test for the head-interpretation pipeline.

Plants heads whose profiles are exactly the stencil, exactly the ACF, exactly
a band-limited operator row, and diffuse noise, then checks the gate, floor,
and selection recover the planted labels. Run this before touching real
checkpoints: if the pipeline cannot recover ground truth it constructs
itself, nothing it says about the models is worth reading.

    python selftest.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import attention_common as C
from assign_heads import assign, aggregate

N = 64
HALF_WIDTH = 4
RNG = np.random.default_rng(0)

STENCILS = {
    ("heat", 0.25): (np.array([0.25, 0.5, 0.25]), np.array([-1, 0, 1])),
    ("lf", 0.25): (np.array([0.625, 0.0, 0.375]), np.array([-1, 0, 1])),
}


def ar1_acf(rho, n):
    d = np.minimum(np.arange(n), n - np.arange(n))
    return rho ** d


def ring_to_profile(ring, offsets):
    p = C.normalise_abs(ring[offsets % N])
    q, _, _ = C.restrict(p, offsets, HALF_WIDTH)
    return C.normalise_abs(q)


def build_predictors():
    store = {}
    for (pde, r), (w, soff) in STENCILS.items():
        key = f"{pde}_r{r:g}"
        kgrid = np.unique(np.linspace(1, N // 8, 8).round().astype(int))
        store[f"{key}/stencil_ring"] = C.stencil_row(w, soff, N)
        store[f"{key}/acf_ring"] = ar1_acf(0.85, N)
        store[f"{key}/spectral_ring"] = np.stack(
            [C.spectral_row(w, soff, N, int(k)) for k in kgrid])
        store[f"{key}/spectral_kgrid"] = kgrid
        store[f"{key}/n_space"] = np.array(N)
    return store


def profile_to_attention(ring_profile):
    """Circulant attention matrix whose diagonal average is the given profile."""
    p = np.abs(ring_profile)
    p = p / p.sum()
    return np.stack([np.roll(p, i) for i in range(N)], axis=0)


def surviving_kmax_index(pred, key, min_structure=0.01):
    """Index of the smallest kmax that stays distinguishable from uniform.

    The planted spectral head must sit inside the identifiable band, otherwise
    the test is asking the pipeline to recover a hypothesis it has correctly
    ruled inadmissible for this window width.
    """
    from assign_heads import predictor_set
    offsets = C.offsets_full(N)
    _, _, spec, kgrid = predictor_set(pred, key, offsets, HALF_WIDTH, min_structure)
    if not len(kgrid):
        raise RuntimeError(
            f"no admissible kmax for {key} at half_width={HALF_WIDTH}; widen the "
            "window or raise the grid floor")
    full = np.asarray(pred[f"{key}/spectral_kgrid"])
    return int(np.where(full == kgrid[0])[0][0])


def synth_run(pde, r, seed, kinds, pred, jitter):
    w, soff = STENCILS[(pde, r)]
    key = f"{pde}_r{r:g}"
    rings = {
        "stencil": np.abs(C.stencil_row(w, soff, N)),
        "acf": np.abs(ar1_acf(0.85, N)),
        "spectral": np.abs(pred[f"{key}/spectral_ring"][surviving_kmax_index(pred, key)]),
        "diffuse": np.ones(N),
    }
    mats = []
    for kind in kinds:
        base = rings[kind] / rings[kind].sum()
        noisy = np.abs(base + jitter * RNG.normal(size=N) * base.max())
        mats.append(profile_to_attention(noisy))
    attn = np.stack(mats).reshape(len(kinds), 1, N, N)  # one head per "layer"
    offsets = C.offsets_full(N)
    return {
        "profile": C.relative_profile(attn, offsets),
        "offsets": offsets,
        "pde": pde, "r": r, "seed": seed, "path": f"synthetic/{pde}_{seed}",
    }


def main():
    store = build_predictors()
    pred = {k: np.asarray(v) for k, v in store.items()}

    kinds = ["stencil", "acf", "spectral", "diffuse"]
    trained_runs, baseline_runs = [], []
    for (pde, r) in STENCILS:
        for seed in range(8):
            trained_runs.append(synth_run(pde, r, seed, kinds, pred, jitter=0.02))
        for seed in range(30):
            baseline_runs.append(
                synth_run(pde, r, 1000 + seed, ["diffuse"] * 4, pred, jitter=0.35))

    from assign_heads import head_records
    trained = head_records(trained_runs, pred, HALF_WIDTH, "js", False)
    baseline = head_records(baseline_runs, pred, HALF_WIDTH, "js", True)

    trained["planted"] = [kinds[l] for l in trained["layer"]]
    df = assign(trained, baseline, gate_pct=5.0, floor_pct=5.0,
                spectral_margin=0.05)

    expected = {"stencil": "stencil", "acf": "acf", "spectral": "spectral",
                "diffuse": "ungated"}
    df["correct"] = [expected[p] == l for p, l in zip(df["planted"], df["label"])]

    confusion = pd.crosstab(df["planted"], df["label"])
    print(confusion.to_string())
    print()
    for planted, grp in df.groupby("planted"):
        print(f"{planted:9s} recovered {grp['correct'].mean():6.1%} "
              f"(n={len(grp)})")
    acc = df["correct"].mean()
    print(f"\noverall {acc:.1%}")

    print("\nfitted kmax on planted spectral heads:",
          sorted(df[df.planted == "spectral"]["spectral_kmax"].unique()))
    print("LF stencil-vs-acf separation (planted stencil heads, lf only):")
    lf = df[(df.pde == "lf") & (df.planted == "stencil")]
    print(f"  d_stencil={lf['d_stencil'].mean():.4f}  d_acf={lf['d_acf'].mean():.4f}")

    assert acc > 0.9, f"recovery too low: {acc:.1%}"
    print("\nself-test passed")


if __name__ == "__main__":
    main()


def jitter_sweep():
    """How much profile noise the selection tolerates before degrading."""
    from assign_heads import head_records
    store = build_predictors()
    pred = {k: np.asarray(v) for k, v in store.items()}
    kinds = ["stencil", "acf", "spectral", "diffuse"]
    expected = {"stencil": "stencil", "acf": "acf", "spectral": "spectral",
                "diffuse": "ungated"}
    print("\njitter  accuracy   labels")
    for jit in [0.02, 0.1, 0.25, 0.5, 1.0]:
        tr, bl = [], []
        for (pde, r) in STENCILS:
            for s in range(8):
                tr.append(synth_run(pde, r, s, kinds, pred, jitter=jit))
            for s in range(30):
                bl.append(synth_run(pde, r, 1000 + s, ["diffuse"] * 4, pred,
                                    jitter=0.35))
        t = head_records(tr, pred, HALF_WIDTH, "js", False)
        b = head_records(bl, pred, HALF_WIDTH, "js", True)
        t["planted"] = [kinds[l] for l in t["layer"]]
        d = assign(t, b, 5.0, 5.0, 0.05)
        acc = np.mean([expected[p] == l for p, l in zip(d["planted"], d["label"])])
        print(f"{jit:6.2f}  {acc:7.1%}   {dict(d['label'].value_counts())}")


if __name__ == "__main__" and "--sweep" in __import__("sys").argv:
    jitter_sweep()
