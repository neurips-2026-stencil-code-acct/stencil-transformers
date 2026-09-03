"""Test whether learned structure tracks r and nu the way the scheme does.

Because the measured statistic and the predicted statistic are the same
functional evaluated on different profiles, a slope of exactly 1 with zero
intercept means the head tracks that hypothesis quantitatively, not merely
monotonically. That is a much sharper test than a correlation, and it is the
reason the statistics in attention_statistics.py were defined to be computed
identically on both sides.

Three models are fitted per (pde, statistic):

    joint          y ~ stencil + acf [+ xcorr]     partial coefficients
    single         y ~ one predictor at a time      falsifiable with few r
    unconstrained  y ~ r                            for reference

The joint model is reported with a hard caveat. With K distinct parameter
values and P predictors, the between-condition fit has K - P - 1 residual
degrees of freedom; at K = 3 and P = 2 that is zero, so the joint model
reproduces the three condition means exactly no matter what the data say. Its
partial coefficients are still identified, but they are not a test. The single
-predictor comparison is what carries evidential weight at K = 3, and
experimental_design.py exists to decide which extra parameter values would restore a
genuine joint test.

Usage
-----
    python parameter_regression.py --stats results/stats.csv --out results
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

from attention_statistics import STATS, SOURCES


def _ols(y: np.ndarray, x: np.ndarray):
    """Least squares with an intercept prepended. Returns (coef, r2)."""
    design = np.column_stack([np.ones(len(y)), x]) if x.size else np.ones((len(y), 1))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    return coef, r2


def _vif(x: np.ndarray) -> np.ndarray:
    """Variance inflation per column; inf when a column is exactly explained."""
    out = []
    for j in range(x.shape[1]):
        others = np.delete(x, j, axis=1)
        if others.shape[1] == 0:
            out.append(1.0)
            continue
        _, r2 = _ols(x[:, j], others)
        out.append(np.inf if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2))
    return np.array(out)


def cluster_bootstrap(df, ycol, xcols, n_boot, rng):
    """Resample seeds with replacement within each condition.

    Heads within a seed share a checkpoint and are not independent, so the
    resampling unit is the seed. Treating heads as independent would shrink
    the intervals by roughly the square root of the head count and make every
    slope look significantly different from both 0 and 1.
    """
    coefs = []
    keys = df[["pde", "r"]].drop_duplicates().to_records(index=False)
    seeds_by_cond = {(p, r): df[(df.pde == p) & (df.r == r)]["seed"].unique()
                     for p, r in keys}
    for _ in range(n_boot):
        parts = []
        for (p, r), seeds in seeds_by_cond.items():
            if len(seeds) == 0:
                continue
            drawn = rng.choice(seeds, size=len(seeds), replace=True)
            sub = df[(df.pde == p) & (df.r == r)]
            parts.extend(sub[sub.seed == s] for s in drawn)
        b = pd.concat(parts)
        coef, _ = _ols(b[ycol].to_numpy(), b[xcols].to_numpy())
        coefs.append(coef)
    return np.array(coefs)


def _near_duplicate(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    """True when two predictor curves agree to within tol after centring."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(float(np.nanmax(np.abs(a))), float(np.nanmax(np.abs(b))), 1e-12)
    return bool(np.nanmax(np.abs((a - a.mean()) - (b - b.mean()))) < tol * scale)


def fit_arm(df: pd.DataFrame, pde: str, stat: str, sources, n_boot, rng):
    sub = df[(df.pde == pde)].dropna(subset=[stat]).copy()
    if sub.empty:
        return [], []

    conds = sub[["r"]].drop_duplicates().sort_values("r")
    n_cond = len(conds)

    usable, dropped = [], []
    for src in sources:
        col = f"pred_{src}_{stat}"
        if col not in sub or sub[col].isna().all():
            dropped.append((src, "not available"))
            continue
        vals = sub.groupby("r")[col].first().to_numpy()
        scale = max(1.0, float(np.nanmax(np.abs(vals))))
        # Scale-aware: a predictor sitting at 1e-17 across all conditions is
        # numerically zero, not a weak signal. An absolute tolerance keeps it,
        # and the regression then divides by its noise-level variance and
        # returns coefficients of order 1e8 with meaningless intervals.
        if np.nanstd(vals) < 1e-8 * scale:
            dropped.append((src, f"constant at {vals[0]:.4g} across all r"))
            continue
        dup = next((u for u in usable
                    if _near_duplicate(sub.groupby("r")[f"pred_{u}_{stat}"]
                                       .first().to_numpy(), vals)), None)
        if dup is not None:
            # On symmetric arms the cross-correlation reduces to the
            # autocorrelation, because the two differ only through the
            # advective phase of the amplification factor. Keeping both would
            # produce an exactly singular design rather than new information.
            dropped.append((src, f"duplicates '{dup}' across all r"))
            continue
        usable.append(src)

    records = []
    xcols = [f"pred_{s}_{stat}" for s in usable]

    # between-condition view: what the joint model can and cannot see
    cond_y = sub.groupby("r")[stat].mean().to_numpy()
    cond_x = sub.groupby("r")[xcols].first().to_numpy() if xcols else np.empty((n_cond, 0))
    resid_df = n_cond - len(usable) - 1

    if xcols and resid_df < 0:
        dropped.append(("joint model",
                        f"underdetermined: {len(usable)} predictors against "
                        f"{n_cond} parameter values"))
        xcols = []

    if xcols:
        vifs = _vif(cond_x) if len(usable) > 1 else np.array([1.0])
        boot = cluster_bootstrap(sub, stat, xcols, n_boot, rng)
        coef, _ = _ols(sub[stat].to_numpy(), sub[xcols].to_numpy())
        _, r2_between = _ols(cond_y, cond_x)
        names = ["intercept"] + usable
        for i, name in enumerate(names):
            lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
            records.append({
                "pde": pde, "statistic": stat, "model": "joint", "term": name,
                "coef": coef[i], "ci_lo": lo, "ci_hi": hi,
                "vif": np.nan if i == 0 else vifs[i - 1],
                "n_conditions": n_cond, "between_resid_df": resid_df,
                "r2_between": r2_between, "n_heads": len(sub),
            })

    for src in usable:
        col = f"pred_{src}_{stat}"
        boot = cluster_bootstrap(sub, stat, [col], n_boot, rng)
        coef, _ = _ols(sub[stat].to_numpy(), sub[[col]].to_numpy())
        _, r2_between = _ols(cond_y, sub.groupby("r")[[col]].first().to_numpy())
        for i, name in enumerate(["intercept", src]):
            lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
            records.append({
                "pde": pde, "statistic": stat, "model": f"single_{src}",
                "term": name, "coef": coef[i], "ci_lo": lo, "ci_hi": hi,
                "vif": 1.0, "n_conditions": n_cond,
                "between_resid_df": n_cond - 2, "r2_between": r2_between,
                "n_heads": len(sub),
            })

    boot = cluster_bootstrap(sub, stat, ["r"], n_boot, rng)
    coef, _ = _ols(sub[stat].to_numpy(), sub[["r"]].to_numpy())
    for i, name in enumerate(["intercept", "r"]):
        lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
        records.append({
            "pde": pde, "statistic": stat, "model": "unconstrained",
            "term": name, "coef": coef[i], "ci_lo": lo, "ci_hi": hi,
            "vif": 1.0, "n_conditions": n_cond, "between_resid_df": n_cond - 2,
            "r2_between": np.nan, "n_heads": len(sub),
        })

    return records, dropped


def make_figure(df, records, path, sources):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdes = sorted(df["pde"].unique())
    stats = ["centrality", "asymmetry"]
    fig, axes = plt.subplots(len(stats), len(pdes),
                             figsize=(3.6 * len(pdes), 3.0 * len(stats)),
                             squeeze=False)
    style = {"stencil": ("#2b6cb0", "-"), "acf": ("#c05621", "--"),
             "xcorr": ("#2f855a", ":")}
    for i, stat in enumerate(stats):
        for j, pde in enumerate(pdes):
            ax = axes[i][j]
            sub = df[df.pde == pde]
            per_seed = sub.groupby(["r", "seed"])[stat].mean().reset_index()
            ax.scatter(per_seed["r"], per_seed[stat], s=8, alpha=0.35,
                       color="#2d3748", zorder=3, label="heads (seed means)")
            m = sub.groupby("r")[stat].mean()
            ax.plot(m.index, m.values, "o-", color="#1a202c", lw=1.6, zorder=4,
                    label="measured")
            for src in sources:
                col = f"pred_{src}_{stat}"
                if col in sub and not sub[col].isna().all():
                    p = sub.groupby("r")[col].first()
                    c, ls = style[src]
                    ax.plot(p.index, p.values, ls, color=c, lw=1.6,
                            label=f"{src} prediction")
            if i == 0:
                ax.set_title(pde)
            if j == 0:
                ax.set_ylabel(stat)
            ax.set_xlabel("r / nu")
    axes[0][-1].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                       frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stats", default="results/stats.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--statistics", default=",".join(STATS))
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--gated-only", action="store_true",
                    help="restrict to labelled heads (drops 'ungated')")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.stats)
    if args.gated_only:
        if df["label"].isna().all():
            sys.exit(
                "--gated-only needs labels; rerun attention_statistics with --head-labels"
            )
        df = df[df["label"].notna() & (df["label"] != "ungated")]
        if df.empty:
            sys.exit("no gated heads survive the filter")

    sources = [s for s in args.sources.split(",") if s]
    stats = [s for s in args.statistics.split(",") if s]
    rng = np.random.default_rng(args.seed)

    records, notes = [], []
    for pde, stat in itertools.product(sorted(df["pde"].unique()), stats):
        recs, dropped = fit_arm(df, pde, stat, sources, args.n_boot, rng)
        records.extend(recs)
        notes.extend((pde, stat, src, why) for src, why in dropped)

    res = pd.DataFrame(records)
    os.makedirs(args.out, exist_ok=True)
    res.to_csv(os.path.join(args.out, "regression.csv"), index=False)
    make_figure(df, res, os.path.join(args.out, "parameter_tracking.png"),
                sources)

    slopes = res[(res.term != "intercept") & (res.model != "unconstrained")]
    print(slopes[["pde", "statistic", "model", "term", "coef", "ci_lo", "ci_hi",
                  "vif", "r2_between"]].round(3).to_string(index=False))

    saturated = res[(res.model == "joint") & (res.between_resid_df <= 0)]
    if len(saturated):
        arms = saturated[["pde", "statistic"]].drop_duplicates()
        print(f"\nwarning: the joint model is saturated for "
              f"{len(arms)} (pde, statistic) arms: with "
              f"{int(saturated['n_conditions'].iloc[0])} parameter values and "
              "2 predictors it reproduces the condition means exactly and "
              "cannot be falsified. Report the single-predictor comparison as "
              "the test, and see experimental_design.py for which extra r values fix "
              "this.", file=sys.stderr)

    high_vif = res[(res.model == "joint") & (res.vif > 10)]
    if len(high_vif):
        print(f"\nwarning: predictors nearly collinear (VIF > 10) in "
              f"{len(high_vif)} joint terms; partial coefficients there are "
              "not separately interpretable.", file=sys.stderr)

    for pde, stat, src, why in notes:
        print(f"note: {pde}/{stat}: dropped '{src}' predictor, {why}",
              file=sys.stderr)

    print(f"\nwrote {args.out}/regression.csv, parameter_tracking.png")


if __name__ == "__main__":
    main()
