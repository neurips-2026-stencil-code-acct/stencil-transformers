"""Build input-output cross-correlation predictors.

The autocorrelation predictor cannot explain asymmetry at all. The circular
spatial ACF of any real field satisfies rho[d] = rho[-d] exactly, so on the
Lax-Friedrichs arm, where asymmetry is the whole signal, the ACF hypothesis
predicts zero asymmetry at every nu regardless of the data. That makes the
asymmetry test one-sided and the acf arm vacuous.

The correct data-side predictor for a next-step task is the cross-correlation
between input u^n and target u^{n+1}:

    c[d] = < u^n_i u^{n+1}_{i+d} >

which is asymmetric whenever the scheme advects, with its peak displaced by
roughly the Courant number. It is a genuine competitor to the stencil on the
LF arm: both predict nonzero, nu-dependent asymmetry, but with different
functional forms. Without it, the LF result would only be able to reject
operator tracking, never to identify what replaced it.

Usage
-----
    python cross_correlation.py --data-root data --out xcorr.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# The attention-analysis modules are the single source of truth; import them from
# ../what_do_heads_learn
# rather than keeping a duplicate copy here (a duplicate attention_common.py used
# to live in this directory and had already drifted from the shared version).
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "what_do_heads_learn"
    ),
)
import attention_common as C
from build_predictors import CONDITION_REGEX, find_conditions, resolve_condition_dir


def cross_correlation(u_in: np.ndarray, u_out: np.ndarray) -> np.ndarray:
    """Mean circular cross-correlation, indexed by offset mod N.

    Each pair is normalised by the geometric mean of its own lag-0
    autocorrelations, so pairs are weighted equally rather than by amplitude,
    matching the convention spatial_acf uses.
    """
    x = np.asarray(u_in, dtype=np.float64)
    y = np.asarray(u_out, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    n = x.shape[-1]
    fx = np.fft.rfft(x, axis=-1)
    fy = np.fft.rfft(y, axis=-1)
    cc = np.fft.irfft((np.conj(fx) * fy), n=n, axis=-1)
    norm = np.sqrt((x ** 2).sum(-1) * (y ** 2).sum(-1))
    good = norm > 0
    cc = cc.reshape(-1, n)[good.reshape(-1)] / norm.reshape(-1)[good.reshape(-1)][:, None]
    if cc.size == 0:
        raise ValueError("no usable (u^n, u^{n+1}) pairs")
    return cc.mean(axis=0)


def load_pairs(data_dir: str, cap: int):
    """Return (u^n, u^{n+1}) stacked over trajectories.

    Requires the trajectory axis to be intact, so a flat snapshot array is not
    enough: consecutive-in-time pairing is the point. Expects (n_traj, n_t, N).
    """
    for name in ["train.npy", "train_u.npy", "u_train.npy",
                 "trajectories.npy", "u.npy"]:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        arr = np.asarray(np.load(path, mmap_mode="r"))
        if arr.ndim < 3:
            raise ValueError(
                f"{path} has shape {arr.shape}; the cross-correlation needs the "
                "time axis intact, i.e. (n_traj, n_t, N). If only flattened "
                "snapshots are stored, regenerate or pass --skip-xcorr and "
                "accept that the acf arm cannot address asymmetry."
            )
        a = arr[:, :-1].reshape(-1, arr.shape[-1])
        b = arr[:, 1:].reshape(-1, arr.shape[-1])
        if a.shape[0] > cap:
            idx = np.sort(np.random.default_rng(0).choice(a.shape[0], cap, False))
            a, b = a[idx], b[idx]
        return np.asarray(a, np.float64), np.asarray(b, np.float64)
    raise FileNotFoundError(f"no trajectory array in {data_dir}")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="xcorr.npz")
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--summary-width", type=int, default=4)
    ap.add_argument("--condition-regex", default=CONDITION_REGEX)
    args = ap.parse_args()

    conditions = find_conditions(args.data_root, args.condition_regex)
    if not conditions:
        sys.exit(f"no condition directories under {args.data_root}")

    store, summary = {}, []
    for pde, r, cond_dir in conditions:
        key = f"{pde}_r{r:g}"
        try:
            # Real data lives per-seed (cond_dir/seed_<N>/train.npy), not
            # flat inside cond_dir -- see resolve_condition_dir's docstring.
            ddir = resolve_condition_dir(cond_dir)
        except FileNotFoundError:
            # Data generation finishes per-condition, not all at once, so an
            # empty placeholder directory is expected, not exceptional.
            print(f"skip {key}: {cond_dir} has no data yet", file=sys.stderr)
            continue
        a, b = load_pairs(ddir, args.pairs)
        cc = cross_correlation(a, b)
        if f"{key}/xcorr_ring" in store:
            print(f"warning: {key} was already built from a different "
                  f"directory; {cond_dir} is overwriting it. Multiple "
                  "directories parse to this (pde, r) -- check for stale or "
                  "mislabeled data.", file=sys.stderr)
        store[f"{key}/xcorr_ring"] = cc
        store[f"{key}/n_space"] = np.array(a.shape[-1])

        # A symmetric window is required here. offsets_full spans -N/2 .. N/2-1,
        # so the Nyquist offset has no positive counterpart and a perfectly even
        # ring would report a spurious negative centroid.
        n = a.shape[-1]
        offs = np.arange(-args.summary_width, args.summary_width + 1)
        ring = np.abs(cc[offs % n])
        centroid = float(np.sum(offs * ring) / np.sum(ring))
        lo, hi = float(np.abs(cc[(n - 1) % n])), float(np.abs(cc[1]))
        asym = (hi - lo) / (hi + lo) if hi + lo > 0 else np.nan
        summary.append((key, asym, centroid, a.shape[0]))

    np.savez_compressed(args.out, **store)
    print(f"{'condition':<14} {'asymmetry':>10} {'centroid':>9} {'pairs':>8}")
    for key, asym, cen, npairs in summary:
        print(f"{key:<14} {asym:>10.4f} {cen:>9.4f} {npairs:>8d}")
    print(f"\nwrote {args.out}")
    print("Nonzero asymmetry on the advection arm confirms the predictor "
          "carries directional information the ACF cannot; the heat and wave "
          "arms should sit near zero, which doubles as a correctness check.")


if __name__ == "__main__":
    main()
