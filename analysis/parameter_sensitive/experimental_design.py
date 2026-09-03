"""Experimental-design stage (run first): which parameter values make the joint test possible?

At three r values the joint regression is saturated and cannot be falsified.
Adding parameter values is cheap relative to what it buys, but only some
additions help: if the stencil-implied and data-implied statistic curves stay
nearly parallel over the values chosen, the two predictors remain collinear
and their partial coefficients stay uninterpretable no matter how many seeds
are trained.

This script answers the question before any compute is spent. It simulates
the schemes directly on band-limited random initial conditions, which is
free compared with training, derives both predictor curves over a fine
candidate grid, and greedily selects the additions that most reduce the
variance of the partial slope estimates (A-optimality on the slope block of
(X'X)^-1).

Validate first. --validate compares the simulated autocorrelation against the
empirical one in predictors.npz at the conditions already generated; if they
disagree, the scheme implementation here does not match the generator and the
recommendation should not be trusted.

Usage
-----
    python experimental_design.py --validate --predictors predictors.npz \
        --ic-spectrum-from data/heat_r0.25
    python experimental_design.py --pde heat --have 0.1,0.25,0.4 \
        --candidates 0.05,0.15,0.2,0.3,0.35,0.45 --add 2 \
        --ic-spectrum-from data/heat_r0.25
"""

from __future__ import annotations

import argparse
import itertools
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
from attention_statistics import shape_statistics

SCHEMES = {"heat", "lf", "wave"}


def step(u, u_prev, pde, r):
    left, right = np.roll(u, 1, -1), np.roll(u, -1, -1)
    if pde == "heat":
        return u + r * (right - 2 * u + left), u
    if pde == "lf":
        return 0.5 * (right + left) - 0.5 * r * (right - left), u
    if pde == "wave":
        return 2 * u - u_prev + r * (right - 2 * u + left), u
    raise ValueError(pde)


def stencil_of(pde, r):
    """Weights at offsets (-1, 0, +1), matching the generators' conventions."""
    if pde == "heat":
        return np.array([r, 1 - 2 * r, r])
    if pde == "lf":
        return np.array([(1 + r) / 2, 0.0, (1 - r) / 2])
    if pde == "wave":
        return np.array([r, 2 - 2 * r, r])
    raise ValueError(pde)


def ic_spectrum(data_dir: str | None, n: int, k0: float, n_traj: int, rng):
    """Mean power spectrum of the initial conditions, measured or synthetic."""
    if data_dir:
        for name in ["train.npy", "trajectories.npy", "u.npy", "test.npy"]:
            path = os.path.join(data_dir, name)
            if os.path.exists(path):
                arr = np.asarray(np.load(path, mmap_mode="r"))
                u0 = arr[:, 0] if arr.ndim >= 3 else arr[:n_traj]
                u0 = np.asarray(u0, np.float64)
                u0 = u0 - u0.mean(-1, keepdims=True)
                return np.abs(np.fft.rfft(u0, axis=-1)).mean(0)
        raise FileNotFoundError(f"no array in {data_dir}")
    k = np.fft.rfftfreq(n, 1.0 / n)
    return np.exp(-((k / k0) ** 2))


def simulate(pde, r, amp, n_traj, n_t, rng):
    n = 2 * (len(amp) - 1)
    phase = rng.normal(size=(n_traj, len(amp))) + 1j * rng.normal(size=(n_traj, len(amp)))
    u = np.fft.irfft(amp[None, :] * phase, n=n, axis=-1)
    u_prev = u.copy()
    snaps, pairs = [u.copy()], []
    for _ in range(n_t - 1):
        u, u_prev = step(u, u_prev, pde, r)
        if not np.all(np.isfinite(u)) or np.abs(u).max() > 1e12:
            return None, None
        pairs.append((snaps[-1], u.copy()))
        snaps.append(u.copy())
    return np.concatenate(snaps, 0), pairs


def curves(pde, r, amp, offsets, half_width, n_traj, n_t, rng):
    """Stencil-implied and data-implied statistics at one parameter value."""
    from cross_correlation import cross_correlation

    n = 2 * (len(amp) - 1)
    sten = C.normalise_abs(C.stencil_row(stencil_of(pde, r), [-1, 0, 1], n)[offsets % n])
    sten, woff, _ = C.restrict(sten, offsets, half_width)

    snaps, pairs = simulate(pde, r, amp, n_traj, n_t, rng)
    if snaps is None:
        return None
    acf = C.spatial_acf(snaps)
    a = np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])
    xc = cross_correlation(*a)

    def _w(ring):
        p = C.normalise_abs(np.asarray(ring)[offsets % n])
        q, _, _ = C.restrict(p, offsets, half_width)
        return C.normalise_abs(q)

    return {
        "stencil": shape_statistics(C.normalise_abs(sten), woff),
        "acf": shape_statistics(_w(acf), woff),
        "xcorr": shape_statistics(_w(xc), woff),
    }


def cross_slope(x_true: np.ndarray, x_fit: np.ndarray) -> float:
    """Slope obtained by fitting predictor `fit` to data generated by `true`.

    This is the metric that actually decides whether the single-predictor
    comparison can tell the hypotheses apart, and it is not the same question
    as collinearity. Two curves can correlate at 0.99 and still be trivially
    distinguishable if one is fifteen times the other or points the other way:
    fitting the wrong one then returns a slope nowhere near 1, and the
    slope-equals-1 test separates them cleanly. Correlation only governs
    whether the *joint* partial coefficients are separately identified.
    """
    xc = x_fit - x_fit.mean()
    var = float(xc @ xc)
    if var <= 0:
        return np.nan
    return float((xc @ (x_true - x_true.mean())) / var)


def a_optimality(x: np.ndarray) -> float:
    """Summed variance of the slope estimates, up to noise scale. Lower is better."""
    design = np.column_stack([np.ones(len(x)), x])
    if len(x) <= design.shape[1]:
        return np.inf
    try:
        cov = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return np.inf
    return float(np.trace(cov[1:, 1:]))


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--pde", default="heat", choices=sorted(SCHEMES))
    ap.add_argument("--statistic", default=None,
                    help="default: asymmetry for lf, centrality otherwise")
    ap.add_argument("--sources", default="stencil,xcorr",
                    help="the two predictors whose separation is being designed")
    ap.add_argument("--have", default="0.1,0.25,0.4")
    ap.add_argument("--candidates", default="")
    ap.add_argument("--add", type=int, default=2)
    ap.add_argument("--n-space", type=int, default=64)
    ap.add_argument("--n-traj", type=int, default=64)
    ap.add_argument("--n-t", type=int, default=200)
    ap.add_argument("--half-width", type=int, default=4)
    ap.add_argument("--k0", type=float, default=6.0)
    ap.add_argument("--ic-spectrum-from", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--predictors", default="predictors.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    amp = ic_spectrum(args.ic_spectrum_from, args.n_space, args.k0, args.n_traj, rng)
    n = 2 * (len(amp) - 1)
    offsets = C.offsets_full(n)
    stat = args.statistic or ("asymmetry" if args.pde == "lf" else "centrality")
    sources = [s for s in args.sources.split(",") if s]

    if args.validate:
        pred = np.load(args.predictors)
        print(f"{'condition':<14} {'JS(sim, empirical acf)':>24}")
        ok = True
        for key in sorted({k.split("/")[0] for k in pred.files}):
            pde, rs = key.rsplit("_r", 1)
            if pde not in SCHEMES:
                continue
            c = curves(pde, float(rs), amp, offsets, args.half_width,
                       args.n_traj, args.n_t, rng)
            if c is None:
                print(f"{key:<14} {'unstable':>24}")
                continue
            emp = C.normalise_abs(np.asarray(pred[f"{key}/acf_ring"])[offsets % n])
            emp, woff, _ = C.restrict(emp, offsets, args.half_width)
            sim_ring = c["acf"]
            d = abs(sim_ring["width"] - shape_statistics(C.normalise_abs(emp), woff)["width"])
            ok &= d < 0.5
            print(f"{key:<14} {d:>24.4f}")
        print("\nvalues are |simulated width - empirical width| in grid units.")
        if not ok:
            print("warning: simulation does not reproduce the generator's "
                  "autocorrelation; the scheme or IC spectrum here differs from "
                  "yours and the recommendation below would be unreliable.",
                  file=sys.stderr)
        return

    have = [float(v) for v in args.have.split(",") if v]
    cands = [float(v) for v in args.candidates.split(",") if v]

    rows = {}
    for r in sorted(set(have + cands)):
        c = curves(args.pde, r, amp, offsets, args.half_width,
                   args.n_traj, args.n_t, rng)
        rows[r] = None if c is None else np.array([c[s][stat] for s in sources])

    print(f"{args.pde}, statistic = {stat}")
    header = f"{'r':>7} " + " ".join(f"{s:>12}" for s in sources) + "   status"
    print(header)
    for r, v in rows.items():
        tag = "have" if r in have else "candidate"
        if v is None:
            print(f"{r:>7.3g} {'unstable':>12}{'':>13}   {tag}")
        else:
            print(f"{r:>7.3g} " + " ".join(f"{x:>12.4f}" for x in v) + f"   {tag}")

    usable = {r: v for r, v in rows.items() if v is not None and np.all(np.isfinite(v))}
    have = [r for r in have if r in usable]
    cands = [r for r in cands if r in usable]

    base = a_optimality(np.array([usable[r] for r in have]))
    print(f"\ncurrent design: {len(have)} values, residual df "
          f"{len(have) - len(sources) - 1}, A-optimality "
          f"{base if np.isfinite(base) else float('inf'):.4g}")
    if not np.isfinite(base):
        print("  (saturated or singular: the joint model is not testable as is)")

    chosen = list(have)
    for step_i in range(args.add):
        best, best_v = None, np.inf
        for r in cands:
            if r in chosen:
                continue
            v = a_optimality(np.array([usable[x] for x in chosen + [r]]))
            if v < best_v:
                best, best_v = r, v
        if best is None:
            break
        chosen.append(best)
        print(f"  + r = {best:g}  ->  A-optimality {best_v:.4g}, residual df "
              f"{len(chosen) - len(sources) - 1}")

    if len(chosen) > len(have):
        print(f"\nrecommended additions: "
              f"{', '.join(f'{r:g}' for r in chosen if r not in have)}")

    x = np.array([usable[r] for r in chosen])
    corr = np.corrcoef(x.T)[0, 1] if x.shape[1] == 2 else np.nan
    print(f"\ndesign over {len(chosen)} values: predictor correlation "
          f"{corr:+.4f}")

    if x.shape[1] == 2:
        signs_differ = np.all(np.sign(x[:, 0]) != np.sign(x[:, 1]))
        s01 = cross_slope(x[:, 0], x[:, 1])
        s10 = cross_slope(x[:, 1], x[:, 0])
        print(f"cross-slopes: data from {sources[0]} fitted with {sources[1]} "
              f"gives {s01:+.3f}; the reverse gives {s10:+.3f} (1.0 would mean "
              "the wrong hypothesis is indistinguishable from the right one)")
        if signs_differ:
            print(f"the two predictions have opposite sign at every value, so "
                  f"the sign of measured {stat} alone separates them")

        if abs(corr) > 0.95:
            print("\nnote: correlation above 0.95 blocks the JOINT partial "
                  "coefficients; report them as unidentified.", file=sys.stderr)
            if min(abs(s01 - 1), abs(s10 - 1)) > 0.5:
                print("the single-predictor comparison is unaffected and "
                      "remains the test for this arm: the cross-slopes are far "
                      "from 1, so fitting the wrong hypothesis is visibly "
                      "wrong even though the curves are collinear.",
                      file=sys.stderr)
            else:
                print("and the cross-slopes are near 1, so the single-predictor "
                      "comparison cannot separate them either. This arm cannot "
                      "distinguish the hypotheses on this statistic at any "
                      "sample size; choose a different statistic or widen the "
                      "parameter range.", file=sys.stderr)


if __name__ == "__main__":
    main()
