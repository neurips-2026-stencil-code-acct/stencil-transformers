"""Head-assignment stage: gated three-way model selection over attention heads.

The pipeline is deliberately a *selection* rather than three significance
tests. Raw signed correlation answered "is this head distinguishable from
noise", which is why it returned 2/84 cells after BH correction (Benjamini &
Hochberg, 1995, "Controlling the false discovery rate", JRSS-B 57(1); see
info.md References). The question
The attention analysis asks "among heads that attend distinctively at all, which of
three profiles is this one closest to".

Two guards, both calibrated against the random-init baseline rather than an
arbitrary constant:

  gate   a head is judged only if its full-profile entropy falls below the
         --gate-pct percentile of baseline head entropies for the same
         condition. Diffuse heads are reported as 'ungated', not dropped.

  floor  a gated head is labelled only if its best divergence beats the
         --floor-pct percentile of baseline heads' best divergence. Otherwise
         'no_fit'.

Spectral carries one fitted parameter (kmax) and its family nests the stencil,
so it must win by --spectral-margin in relative terms.

Usage
-----
    python assign_heads.py --trained profiles/trained --baseline profiles/baseline \
        --predictors predictors.npz --out results
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import attention_common as C

LABELS = ["stencil", "acf", "spectral", "acf_or_spectral",
          "no_fit", "ungated"]
COLOURS = {
    "stencil": "#2b6cb0",
    "acf": "#c05621",
    "spectral": "#2f855a",
    "acf_or_spectral": "#975a16",
    "no_fit": "#a0aec0",
    "ungated": "#e2e8f0",
}


def load_profiles(directory: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        z = np.load(path, allow_pickle=False)
        out.append({
            "profile": z["profile"],           # (L, H, P)
            "offsets": z["offsets"],
            "pde": str(z["pde"]),
            "r": float(z["r"]),
            "seed": int(z["seed"]),
            "path": path,
        })
    if not out:
        sys.exit(f"no profiles in {directory}")
    return out


def predictor_set(pred, key: str, offsets: np.ndarray, half_width: int,
                  min_structure: float = 0.01):
    """Windowed, renormalised predictor profiles for one condition.

    The spectral grid is pruned from below. A band-limited operator row has
    wavelength ~N/kmax, so at small kmax it is essentially flat across a narrow
    window: the predictor collapses onto the uniform distribution and stops
    being a hypothesis at all. Left unguarded this is actively harmful, because
    uniform random-init heads then fit 'spectral' almost exactly and the label
    becomes a synonym for 'unstructured'. kmax values whose windowed profile
    sits within min_structure of uniform are dropped, which ties the usable
    bandwidth range to the window width instead of leaving it a free choice.
    """
    ring_idx = offsets % int(pred[f"{key}/n_space"])

    def _win(ring):
        p = C.normalise_abs(np.asarray(ring)[ring_idx])
        q, _, _ = C.restrict(p, offsets, half_width)
        return C.normalise_abs(q)

    stencil = _win(pred[f"{key}/stencil_ring"])
    acf = _win(pred[f"{key}/acf_ring"])
    kgrid = np.asarray(pred[f"{key}/spectral_kgrid"])
    spec = np.stack([_win(row) for row in pred[f"{key}/spectral_ring"]], axis=0)

    uniform = np.full(spec.shape[-1], 1.0 / spec.shape[-1])
    keep = np.array([C.js_divergence(s, uniform) >= min_structure for s in spec])
    return stencil, acf, spec[keep], kgrid[keep]


def head_records(runs, pred, half_width, divergence, is_baseline,
                 min_structure=0.01):
    fn = C.DIVERGENCES[divergence]
    rows = []
    for run in runs:
        key = f"{run['pde']}_r{run['r']:g}"
        if f"{key}/stencil_ring" not in pred:
            print(f"warning: no predictors for {key}, skipping {run['path']}",
                  file=sys.stderr)
            continue
        offsets = run["offsets"]
        sten, acf, spec, kgrid = predictor_set(pred, key, offsets, half_width,
                                               min_structure)
        prof = run["profile"].astype(np.float64)          # (L, H, P)
        ent = C.normalised_entropy(prof)                  # full-profile gate stat
        win, _, mass = C.restrict(prof, offsets, half_width)

        n_layers, n_heads = prof.shape[0], prof.shape[1]
        for l in range(n_layers):
            for h in range(n_heads):
                p = win[l, h]
                d_sten = fn(p, sten)
                d_acf = fn(p, acf)
                if len(spec):
                    d_spec_all = np.array([fn(p, s) for s in spec])
                    j = int(np.argmin(d_spec_all))
                    d_spec, kfit = float(d_spec_all[j]), int(kgrid[j])
                else:
                    d_spec, kfit = np.inf, -1
                rows.append({
                    "pde": run["pde"], "r": run["r"], "seed": run["seed"],
                    "layer": l, "head": h,
                    "is_baseline": is_baseline,
                    "entropy": float(ent[l, h]),
                    "peak": float(prof[l, h].max()),
                    "window_mass": float(mass[l, h]),
                    "d_stencil": d_sten,
                    "d_acf": d_acf,
                    "d_spectral": d_spec,
                    "spectral_kmax": kfit,
                })
    return pd.DataFrame(rows)


PREDICTORS = ["stencil", "acf", "spectral"]


def _robust_scale(x: np.ndarray):
    """Median and MAD-derived sigma; falls back to std for degenerate MADs.

    1.4826 is the consistency constant for the normal distribution; see
    Rousseeuw & Croux, 1993, "Alternatives to the median absolute
    deviation", JASA 88(424). See info.md References.
    """
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if mad <= 0:
        mad = float(np.std(x)) or 1.0
    return med, mad


def assign(trained: pd.DataFrame, baseline: pd.DataFrame, gate_pct, floor_pct,
           spectral_margin, merge_unidentifiable=()) -> pd.DataFrame:
    """Label heads by calibrated fit, one null per predictor.

    Each predictor gets its own baseline divergence distribution, and a head's
    fit is scored as z = (d - median_base) / sigma_base. This matters more than
    it looks: the three predictors are not equally easy to fit by chance. A
    band-limited spectral profile at low kmax is broad and nearly uniform, so
    random attention scores a small raw divergence against it, while the
    3-point stencil is a hard target that even a correct head cannot match
    closely in absolute terms. Comparing raw divergences across predictors
    therefore has a built-in bias toward the broadest hypothesis. Calibrating
    each against its own null removes it, and also absorbs the advantage
    spectral gets from fitting kmax, since baseline heads minimise over the
    same grid.
    """
    df = trained.copy()

    stats, gate_thr_by_cond, floor_thr_by_cond = {}, {}, {}
    for (pde, r), grp in baseline.groupby(["pde", "r"]):
        per_pred = {p: _robust_scale(grp[f"d_{p}"].to_numpy()) for p in PREDICTORS}
        stats[(pde, r)] = per_pred
        gate_thr_by_cond[(pde, r)] = float(np.percentile(grp["entropy"], gate_pct))
        floor_thr_by_cond[(pde, r)] = {
            p: float(np.percentile(
                (grp[f"d_{p}"].to_numpy() - per_pred[p][0]) / per_pred[p][1],
                floor_pct))
            for p in PREDICTORS
        }

    for p in PREDICTORS:
        df[f"z_{p}"] = np.nan

    labels, gate_thr, floor_thr = [], [], []
    for idx, row in df.iterrows():
        cond = (row["pde"], row["r"])
        if cond not in stats:
            gate_thr.append(np.nan)
            floor_thr.append(np.nan)
            labels.append("ungated")
            continue
        z = {}
        for p in PREDICTORS:
            med, sig = stats[cond][p]
            z[p] = (row[f"d_{p}"] - med) / sig
            df.at[idx, f"z_{p}"] = z[p]

        g, f = gate_thr_by_cond[cond], floor_thr_by_cond[cond]
        gate_thr.append(g)
        floor_thr.append(min(f.values()))

        # Selection uses the raw fit: it compares one head against three
        # targets, so the scale is common and the comparison is meaningful.
        # Calibration is reserved for validity, below, where the comparison is
        # across heads against a fixed target and the nulls genuinely differ.
        local_pred = ("stencil" if row["d_stencil"] <= row["d_acf"] else "acf")
        if row["d_spectral"] < (1.0 - spectral_margin) * row[f"d_{local_pred}"]:
            choice = "spectral"
        else:
            choice = local_pred

        if row["entropy"] >= g:
            labels.append("ungated")
        elif z[choice] >= f[choice]:
            labels.append("no_fit")
        else:
            labels.append(choice)

    df["label"] = labels
    if merge_unidentifiable:
        for cond in merge_unidentifiable:
            m = (df["pde"] == cond[0]) & (df["r"] == cond[1]) & \
                df["label"].isin(["acf", "spectral"])
            df.loc[m, "label"] = "acf_or_spectral"
    df["best_z"] = df[[f"z_{p}" for p in PREDICTORS]].min(axis=1)
    df["gate_threshold"] = gate_thr
    df["floor_threshold"] = floor_thr
    return df


def predictor_separation(pred, conditions, offsets_by_cond, half_width, divergence,
                         min_structure=0.01):
    """Pairwise divergence between the three predictor profiles.

    The three-way selection is only identifiable to the extent the predictors
    differ. In the Lax-Friedrichs arm they should separate sharply, because a
    zero centre weight is not something an autocorrelation profile produces.
    In the heat arm at small r the stencil approaches a delta while the ACF
    stays broad, so they also separate. The pair to watch is acf/spectral:
    a band-limited operator row and a smooth autocorrelation can be close, and
    where they are, the acf-vs-spectral split should not be interpreted.
    """
    fn = C.DIVERGENCES[divergence]
    rows = []
    for key in conditions:
        offsets = offsets_by_cond[key]
        sten, acf, spec, kgrid = predictor_set(pred, key, offsets, half_width,
                                               min_structure)
        if len(spec):
            best = spec[int(np.argmin([fn(acf, s) for s in spec]))]
            d_ss = min(fn(sten, s) for s in spec)
            d_as = fn(acf, best)
        else:
            d_ss = d_as = np.nan
        rows.append({
            "condition": key,
            "n_spectral_kmax": len(spec),
            "kmax_min": int(kgrid.min()) if len(kgrid) else -1,
            "kmax_max": int(kgrid.max()) if len(kgrid) else -1,
            "d_stencil_acf": fn(sten, acf),
            "d_stencil_spectral": d_ss,
            "d_acf_spectral": d_as,
        })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    counts = (df.groupby(["pde", "r", "layer", "label"]).size()
                .unstack("label").reindex(columns=LABELS).fillna(0))
    frac = counts.div(counts.sum(axis=1), axis=0)
    return frac.reset_index()


def make_figure(frac: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdes = sorted(frac["pde"].unique())
    fig, axes = plt.subplots(1, len(pdes), figsize=(4.2 * len(pdes), 3.4),
                             sharey=True, squeeze=False)
    for ax, pde in zip(axes[0], pdes):
        sub = frac[frac["pde"] == pde].sort_values(["r", "layer"])
        x = np.arange(len(sub))
        bottom = np.zeros(len(sub))
        for lab in LABELS:
            vals = sub[lab].to_numpy()
            ax.bar(x, vals, bottom=bottom, color=COLOURS[lab], label=lab,
                   width=0.82, edgecolor="white", linewidth=0.4)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:g}\nL{l}" for r, l in
                            zip(sub["r"], sub["layer"])], fontsize=7)
        ax.set_title(pde)
        ax.set_ylim(0, 1)
        ax.set_xlabel("r / layer")
    axes[0][0].set_ylabel("fraction of heads")
    axes[0][-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                       frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")


def kmax_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Fitted-kmax counts for heads labelled 'spectral', one row per
    (pde, r, spectral_kmax).

    This is the diagnostic the spectral hypothesis needs to be reportable
    rather than a black box: a head fit at the low end of the kmax grid is a
    genuine band-limited/low-pass computation, while a head fit at the top
    of the grid is nesting toward the stencil (spectral_row's docstring:
    kmax -> N/2 recovers the stencil exactly) and its 'spectral' label
    should be read with that caveat even though it already had to beat
    --spectral-margin to be assigned at all.
    """
    spec = df[df["label"] == "spectral"]
    if spec.empty:
        return pd.DataFrame(columns=["pde", "r", "spectral_kmax", "count"])
    counts = (spec.groupby(["pde", "r", "spectral_kmax"]).size()
                  .rename("count").reset_index())
    return counts.sort_values(["pde", "r", "spectral_kmax"]).reset_index(drop=True)


def make_kmax_figure(df: pd.DataFrame, path: str) -> bool:
    """Histogram of fitted kmax for spectral-labelled heads, one panel per PDE.

    Returns False (and writes nothing) if no head was labelled 'spectral' --
    e.g. because every condition merged into acf_or_spectral, which is an
    expected outcome documented in info.md, not a failure of this function.
    """
    spec = df[df["label"] == "spectral"]
    if spec.empty:
        return False

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdes = sorted(spec["pde"].unique())
    fig, axes = plt.subplots(1, len(pdes), figsize=(4.2 * len(pdes), 3.0),
                             squeeze=False)
    for ax, pde in zip(axes[0], pdes):
        sub = spec[spec["pde"] == pde]
        lo, hi = int(sub["spectral_kmax"].min()), int(sub["spectral_kmax"].max())
        bins = np.arange(lo, hi + 2) - 0.5
        for r, grp in sub.groupby("r"):
            ax.hist(grp["spectral_kmax"], bins=bins, alpha=0.6, label=f"r={r:g}")
        ax.set_title(pde)
        ax.set_xlabel("fitted kmax")
        ax.legend(fontsize=7, frameon=False)
    axes[0][0].set_ylabel("# heads labelled spectral")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    return True


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--trained", default="profiles/trained")
    ap.add_argument("--baseline", default="profiles/baseline")
    ap.add_argument("--predictors", default="predictors.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--half-width", type=int, default=4,
                    help="offsets |d| <= this are compared; must exceed 1 so the "
                         "spectral profile's tails are distinguishable from the "
                         "stencil's exact zeros")
    ap.add_argument("--divergence", default="js", choices=sorted(C.DIVERGENCES))
    ap.add_argument("--gate-pct", type=float, default=5.0)
    ap.add_argument("--floor-pct", type=float, default=5.0)
    ap.add_argument("--min-structure", type=float, default=0.01,
                    help="drop spectral kmax values whose windowed profile is "
                         "within this JS divergence of uniform")
    ap.add_argument("--spectral-margin", type=float, default=0.05,
                    help="relative margin by which the band-limited fit must beat "
                         "the best local hypothesis, since its family nests the "
                         "stencil and carries a fitted kmax")
    args = ap.parse_args()

    if args.half_width < 2:
        sys.exit("--half-width must be >= 2 for the three hypotheses to separate")

    os.makedirs(args.out, exist_ok=True)
    pred = np.load(args.predictors)

    trained = head_records(load_profiles(args.trained), pred, args.half_width,
                           args.divergence, False, args.min_structure)
    baseline = head_records(load_profiles(args.baseline), pred, args.half_width,
                            args.divergence, True, args.min_structure)

    runs = load_profiles(args.trained)
    offsets_by_cond = {f"{r['pde']}_r{r['r']:g}": r["offsets"] for r in runs}
    sep = predictor_separation(pred, sorted(offsets_by_cond), offsets_by_cond,
                               args.half_width, args.divergence,
                               args.min_structure)
    sep.to_csv(os.path.join(args.out, "predictor_separation.csv"), index=False)
    print("predictor separation (higher = more identifiable)")
    print(sep.round(4).to_string(index=False))
    weak = sep[sep["d_acf_spectral"] < args.min_structure]
    if len(weak):
        print(f"\nnote: acf and spectral are not separable in "
              f"{', '.join(weak['condition'])}; heads matching either are "
              "reported as acf_or_spectral there", file=sys.stderr)
    print()

    med_mass = trained.groupby(["pde", "r"])["window_mass"].median()
    if (med_mass < 0.2).any():
        bad = ", ".join(f"{p}_r{r:g}" for (p, r) in med_mass[med_mass < 0.2].index)
        print(f"warning: median in-window attention mass below 0.2 for {bad}. "
              "Either these heads genuinely attend far outside the window, or "
              "the profile axis is misaligned with the saved offsets array "
              "(check that profile[..., i] corresponds to offsets[i]).",
              file=sys.stderr)

    unident = tuple(
        (c.rsplit("_r", 1)[0], float(c.rsplit("_r", 1)[1]))
        for c in sep.loc[sep["d_acf_spectral"] < args.min_structure, "condition"]
    )
    df = assign(trained, baseline, args.gate_pct, args.floor_pct,
                args.spectral_margin, merge_unidentifiable=unident)
    frac = aggregate(df)

    df.to_csv(os.path.join(args.out, "heads.csv"), index=False)
    baseline.to_csv(os.path.join(args.out, "baseline_heads.csv"), index=False)
    frac.to_csv(os.path.join(args.out, "label_fractions.csv"), index=False)
    make_figure(frac, os.path.join(args.out, "head_labels.png"))

    kdist = kmax_distribution(df)
    kdist.to_csv(os.path.join(args.out, "spectral_kmax_distribution.csv"), index=False)
    wrote_kmax_fig = make_kmax_figure(
        df, os.path.join(args.out, "spectral_kmax_distribution.png")
    )
    if not wrote_kmax_fig:
        print("\nnote: no heads labelled 'spectral' (possibly all merged into "
              "acf_or_spectral); spectral_kmax_distribution.png not written",
              file=sys.stderr)

    overall = (df.groupby(["pde", "label"]).size()
                 .unstack("label").reindex(columns=LABELS).fillna(0))
    overall = overall.div(overall.sum(axis=1), axis=0)
    print(overall.round(3).to_string())
    print(f"\ngated heads: {(df['label'].isin(['stencil','acf','spectral'])).mean():.1%}")
    print(f"wrote {args.out}/heads.csv, label_fractions.csv, head_labels.png, "
          f"spectral_kmax_distribution.csv")


if __name__ == "__main__":
    main()
