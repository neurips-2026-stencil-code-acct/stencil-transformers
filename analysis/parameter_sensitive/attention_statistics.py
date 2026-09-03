"""Reduce each head and predictor to bounded scalar statistics.

Four statistics, all computed identically on attention profiles and on
predictor profiles so the comparison needs no closed-form algebra and
inherits the |w| and renormalisation conventions from the attention analysis:

    centrality  p[0] / (p[-1] + p[0] + p[+1])        in [0, 1]
    asymmetry   (p[+1] - p[-1]) / (p[-1] + p[+1])    in [-1, 1]
    width       sqrt(sum_d p[d] d^2)                  window-dependent
    shift       sum_d p[d] d                          window-dependent

Bounded forms are used deliberately in place of log ratios. The
Lax-Friedrichs stencil has an exact zero centre weight, so any log-ratio
statistic is -inf for the stencil predictor on the arm that matters most, and
the regression downstream would have to drop or fudge it. Under these
definitions the predicted values stay finite and, satisfyingly, linear:

    heat stencil [r, 1-2r, r]      centrality = 1 - 2r,  asymmetry = 0
    wave spatial [r, 2-2r, r]      centrality = 1 - r,   asymmetry = 0
    LF stencil [(1+nu)/2, 0, ...]  centrality = 0,       asymmetry = -nu

so the stencil hypothesis makes a sharp, parameter-free prediction about the
slope, not merely about monotonicity.

Usage
-----
    python attention_statistics.py --profiles profiles/trained --predictors predictors.npz \
        --xcorr xcorr.npz --head-labels ../what_do_heads_learn/results/heads.csv \
        --out results/stats.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# The attention-analysis modules are the single source of truth; import them from
# ../what_do_heads_learn
# rather than keeping a duplicate copy here (a duplicate attention_common.py used
# to live in this directory and had already drifted from the real one).
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "what_do_heads_learn"
    ),
)
import attention_common as C
from assign_heads import load_profiles, predictor_set

STATS = ["centrality", "asymmetry", "width", "shift"]
SOURCES = ["stencil", "acf", "xcorr"]


def shape_statistics(p: np.ndarray, offsets: np.ndarray) -> dict:
    """Bounded shape statistics of a windowed, normalised offset profile."""
    p = np.asarray(p, dtype=np.float64)
    offsets = np.asarray(offsets)
    idx = {int(d): i for i, d in enumerate(offsets)}
    if not {-1, 0, 1} <= idx.keys():
        raise ValueError("window must contain offsets -1, 0, +1")
    lo, ct, hi = p[idx[-1]], p[idx[0]], p[idx[1]]

    triple = lo + ct + hi
    flank = lo + hi
    return {
        "centrality": float(ct / triple) if triple > 0 else np.nan,
        "asymmetry": float((hi - lo) / flank) if flank > 0 else np.nan,
        "width": float(np.sqrt(np.sum(p * offsets ** 2))),
        "shift": float(np.sum(p * offsets)),
    }


def predictor_statistics(pred, xcorr, key, offsets, half_width, min_structure):
    """Statistics of each predictor profile for one condition.

    The spectral family is deliberately absent. It carries a fitted kmax, so
    its implied statistic is not a fixed function of r and cannot enter a
    regression whose whole point is that the right-hand side is pinned by the
    scheme. The attention analysis already reports whether spectral is separable at all.
    """
    sten, acf, _, _ = predictor_set(pred, key, offsets, half_width, min_structure)
    woff = np.asarray(offsets)[np.abs(np.asarray(offsets)) <= half_width]
    out = {"stencil": shape_statistics(sten, woff),
           "acf": shape_statistics(acf, woff)}

    if xcorr is not None and f"{key}/xcorr_ring" in xcorr:
        n = int(xcorr[f"{key}/n_space"])
        p = C.normalise_abs(np.asarray(xcorr[f"{key}/xcorr_ring"])[offsets % n])
        q, _, _ = C.restrict(p, offsets, half_width)
        out["xcorr"] = shape_statistics(C.normalise_abs(q), woff)
    else:
        out["xcorr"] = {s: np.nan for s in STATS}
    return out


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--profiles", default="profiles/trained")
    ap.add_argument("--predictors", default="predictors.npz")
    ap.add_argument("--xcorr", default=None)
    ap.add_argument(
        "--head-labels",
        default=None,
        help="attention-analysis heads.csv, used to carry labels through",
    )
    ap.add_argument("--half-width", type=int, default=4)
    ap.add_argument("--min-structure", type=float, default=0.01)
    ap.add_argument("--out", default="results/stats.csv")
    args = ap.parse_args()

    pred = np.load(args.predictors)
    xcorr = np.load(args.xcorr) if args.xcorr else None
    if xcorr is None:
        print("note: no --xcorr given. The acf predictor is exactly even, so "
              "asymmetry results will have no data-side competitor to the "
              "stencil and can only reject, not identify.", file=sys.stderr)

    rows, pred_rows = [], []
    for run in load_profiles(args.profiles):
        key = f"{run['pde']}_r{run['r']:g}"
        if f"{key}/stencil_ring" not in pred:
            print(f"warning: no predictors for {key}, skipping", file=sys.stderr)
            continue
        offsets = run["offsets"]
        prof = run["profile"].astype(np.float64)
        win, woff, mass = C.restrict(prof, offsets, args.half_width)

        pstats = predictor_statistics(pred, xcorr, key, offsets,
                                      args.half_width, args.min_structure)
        if not any(d["condition"] == key for d in pred_rows):
            rec = {"condition": key, "pde": run["pde"], "r": run["r"]}
            for src in SOURCES:
                for s in STATS:
                    rec[f"pred_{src}_{s}"] = pstats[src][s]
            pred_rows.append(rec)

        for l in range(prof.shape[0]):
            for h in range(prof.shape[1]):
                rec = {"pde": run["pde"], "r": run["r"], "seed": run["seed"],
                       "layer": l, "head": h,
                       "window_mass": float(mass[l, h])}
                rec.update(shape_statistics(win[l, h], woff))
                for src in SOURCES:
                    for s in STATS:
                        rec[f"pred_{src}_{s}"] = pstats[src][s]
                rows.append(rec)

    df = pd.DataFrame(rows)
    if args.head_labels and os.path.exists(args.head_labels):
        label_rows = pd.read_csv(args.head_labels)[
            ["pde", "r", "seed", "layer", "head", "label", "entropy"]]
        df = df.merge(
            label_rows,
            on=["pde", "r", "seed", "layer", "head"],
            how="left",
        )
    else:
        df["label"] = np.nan

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    pdf = pd.DataFrame(pred_rows).sort_values(["pde", "r"])
    pdf.to_csv(args.out.replace(".csv", "_predicted.csv"), index=False)

    cols = ["condition"] + [f"pred_{s}_{t}" for s in SOURCES
                            for t in ["centrality", "asymmetry"]]
    print(pdf[cols].round(4).to_string(index=False))
    print()
    print(df.groupby(["pde", "r"])[STATS].mean().round(4).to_string())
    print(f"\nwrote {args.out} ({len(df)} heads)")


if __name__ == "__main__":
    main()
