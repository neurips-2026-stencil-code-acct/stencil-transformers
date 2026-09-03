"""Build per-condition predictor profiles for the three hypotheses.

  stencil   normalised |w| from stencil.npy, zero everywhere else
  acf       empirical spatial autocorrelation of the training inputs
  spectral  row of the true operator with its symbol truncated at kmax,
            evaluated over a grid of kmax and fitted per head downstream

The spectral family nests the stencil at kmax = N/2, so the grid is capped
(--kmax-frac) and the assignment step additionally requires spectral to win
by a margin. The fitted kmax distribution is the reportable diagnostic: a
head that is really doing a compact stencil will fit at the top of the grid,
a genuine low-pass head at the bottom.

Usage
-----
    python build_predictors.py --data-root data --out predictors.npz
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

import attention_common as C


def find_conditions(data_root: str, regex: str):
    out = []
    for path in sorted(glob.glob(os.path.join(data_root, "*"))):
        if not os.path.isdir(path):
            continue
        m = re.search(regex, os.path.basename(path), flags=re.IGNORECASE)
        if not m:
            continue
        pde = C.PDE_ALIASES.get(m.group("pde").lower(), m.group("pde").lower())
        out.append((pde, float(m.group("r")), path))
    return out


CONDITION_REGEX = (
    r"(?P<pde>heat|wave|lf|advection|lax_friedrichs)"
    # Kept identical to attention_common.DEFAULT_RUN_REGEX's pde/r portion -- see the
    # comment there for why both the "_new" token and the doubled separator
    # around (?:r|nu)? are needed to match this project's real directory names.
    r"(?:_new)?[_-]?(?:r|nu)?[_-]?(?P<r>[0-9]*\.?[0-9]+)"
)


def resolve_condition_dir(cond_dir: str) -> str:
    """Find the directory that actually holds stencil.npy/train.npy for a
    condition returned by find_conditions().

    Real data lives per-seed: run_heat.py and run_friedrichs.py both point
    the generator's --save_dir at "<condition>/seed_<N>", so
    "data/heat_new_0.25" itself has no files, only seed_100/, seed_101/, ...
    subdirectories -- each an independent draw of initial conditions, but
    with an identical stencil.npy, since the stencil depends only on r. One
    legacy directory (heat_0.25) has files at both levels; the flat layout
    is preferred there since it needs no further choice among seeds.

    Any one seed's data is fine for these population-level predictors (this
    is different from extract_attention_profiles.py's condition_data_dir, which
    must match a specific checkpoint to its own seed's held-out data).
    """
    if os.path.exists(os.path.join(cond_dir, "stencil.npy")):
        return cond_dir
    for seed_dir in sorted(glob.glob(os.path.join(cond_dir, "seed_*"))):
        if os.path.exists(os.path.join(seed_dir, "stencil.npy")):
            return seed_dir
    raise FileNotFoundError(
        f"no stencil.npy in {cond_dir} or any {cond_dir}/seed_*/ subdirectory"
    )


def load_stencil(data_dir: str, offsets_spec: str):
    path = os.path.join(data_dir, "stencil.npy")
    w = np.load(path).ravel().astype(np.float64)
    if offsets_spec == "auto":
        if len(w) % 2 == 0:
            raise ValueError(
                f"{path} has even length {len(w)}; pass --stencil-offsets "
                "explicitly so the centre is unambiguous"
            )
        half = len(w) // 2
        offs = np.arange(-half, half + 1)
    else:
        offs = np.array([int(t) for t in offsets_spec.split(",")])
        if len(offs) != len(w):
            raise ValueError(f"{len(offs)} offsets for a length-{len(w)} stencil")
    return w, offs


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="predictors.npz")
    ap.add_argument("--n-space", type=int, default=None,
                    help="ring size N; inferred from the input array if omitted")
    ap.add_argument("--acf-snapshots", type=int, default=20000,
                    help="max snapshots sampled for the empirical ACF")
    ap.add_argument("--stencil-offsets", default="auto",
                    help="comma-separated offsets for stencil.npy, e.g. '-1,0,1'. "
                         "Convention: entry k multiplies u_{i+offset_k}.")
    ap.add_argument("--kmax-frac", type=float, default=0.25,
                    help="spectral grid capped at kmax <= frac * N/2")
    ap.add_argument("--kmax-steps", type=int, default=12)
    ap.add_argument("--condition-regex", default=CONDITION_REGEX)
    args = ap.parse_args()

    conditions = find_conditions(args.data_root, args.condition_regex)
    if not conditions:
        sys.exit(f"no condition directories under {args.data_root}")

    store: dict[str, np.ndarray] = {}
    summary = []

    for pde, r, cond_dir in conditions:
        key = f"{pde}_r{r:g}"
        try:
            ddir = resolve_condition_dir(cond_dir)
        except FileNotFoundError:
            # Training/data generation finishes per-condition, not all at
            # once, so an empty placeholder directory (e.g. wave isn't
            # generated at all yet) is an expected, not exceptional, state.
            # Skip it and keep building predictors for whatever is ready.
            print(f"skip {key}: {cond_dir} has no data yet", file=sys.stderr)
            continue
        w, soffs = load_stencil(ddir, args.stencil_offsets)

        # ACF is computed on the same held-out snapshots the profiles use, so
        # the comparison is like-for-like.
        u = _load_snapshots(ddir, args.acf_snapshots)
        n = args.n_space or u.shape[-1]
        if u.shape[-1] != n:
            raise ValueError(f"{ddir}: snapshots have N={u.shape[-1]}, expected {n}")

        acf_ring = C.spatial_acf(u)                       # length N, indexed mod N
        stencil_ring = C.stencil_row(w, soffs, n)

        kmax_cap = max(1, int(round(args.kmax_frac * (n // 2))))
        kgrid = np.unique(np.linspace(1, kmax_cap, args.kmax_steps).round().astype(int))
        spectral_ring = np.stack(
            [C.spectral_row(w, soffs, n, int(k)) for k in kgrid], axis=0
        )

        if f"{key}/stencil_ring" in store:
            # More than one directory parses to the same (pde, r) -- this
            # project has both a legacy and a "_new" data convention for
            # heat, and they are not guaranteed to agree (confirmed: the
            # legacy heat_0.25/stencil.npy on disk actually holds an
            # r=0.1024 stencil despite its name). Silently keeping whichever
            # directory happened to sort last would hide that. This is a
            # data problem, not something to auto-resolve here.
            print(f"warning: {key} was already built from a different "
                  f"directory; {cond_dir} is overwriting it. Multiple "
                  "directories parse to this (pde, r) -- check for stale or "
                  "mislabeled data.", file=sys.stderr)
        store[f"{key}/stencil_ring"] = stencil_ring
        store[f"{key}/acf_ring"] = acf_ring
        store[f"{key}/spectral_ring"] = spectral_ring
        store[f"{key}/spectral_kgrid"] = kgrid
        store[f"{key}/stencil_raw"] = w
        store[f"{key}/stencil_offsets"] = soffs
        store[f"{key}/n_space"] = np.array(n)
        store[f"{key}/n_acf_snapshots"] = np.array(u.shape[0])

        # ACF decay length at |rho| < 1/e, a compact summary for the paper.
        ring_off = np.arange(1, n // 2 + 1)
        decay = next((int(d) for d in ring_off if abs(acf_ring[d]) < np.exp(-1.0)), -1)
        summary.append((key, w, acf_ring[1], decay, kgrid[0], kgrid[-1]))

    np.savez_compressed(args.out, **store)

    print(f"{'condition':<14} {'stencil':<28} {'acf(1)':>8} {'acf 1/e':>8} "
          f"{'kmax grid':>12}")
    for key, w, acf1, decay, k0, k1 in summary:
        sten = "[" + ", ".join(f"{v:.4g}" for v in w) + "]"
        print(f"{key:<14} {sten:<28} {acf1:>8.4f} {decay:>8d} {f'{k0}..{k1}':>12}")
    print(f"\nwrote {args.out} ({len(summary)} conditions)")


def _load_snapshots(data_dir: str, cap: int) -> np.ndarray:
    for name in ["train.npy", "train_u.npy", "u_train.npy",
                 "trajectories.npy", "u.npy", "test.npy"]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            arr = np.load(path, mmap_mode="r")
            arr = np.asarray(arr).reshape(-1, arr.shape[-1])
            if arr.shape[0] > cap:
                idx = np.random.default_rng(0).choice(arr.shape[0], cap, replace=False)
                arr = arr[np.sort(idx)]
            return np.asarray(arr, dtype=np.float64)
    raise FileNotFoundError(f"no trajectory array in {data_dir}")


if __name__ == "__main__":
    main()
