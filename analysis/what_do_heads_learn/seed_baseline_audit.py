"""Audit the final seed-level trained-versus-random attention comparison.

The independent unit is one model initialization. Per-head matrices are first
averaged within each trained or random model, then the 20 trained seed values
are compared with the 30 random-initialization values. Benjamini-Hochberg is
applied across the nine PDE/parameter conditions separately for each metric.

Inputs are the original metric arrays produced by analysis/metrics.py and
compute_random_baselines.py, preserving their exact definitions.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


CONDITION_RE = re.compile(
    r"(?P<pde>heat|wave|lax_friedrichs).*?(?:r_?|new_)(?P<r>0?\.\d+)",
    re.IGNORECASE,
)

FILES = {
    "signed_corr": ("corr_matrix.npy", "random_baseline_corrs.npy"),
    "locality": ("locality_matrix.npy", "random_baseline_locality.npy"),
    "entropy": ("entropy_matrix.npy", "random_baseline_entropy.npy"),
}


def parse_condition(name: str) -> tuple[str, float]:
    m = CONDITION_RE.search(name)
    if not m:
        raise ValueError(f"cannot parse condition directory: {name}")
    pde = "lf" if m.group("pde").lower() == "lax_friedrichs" else m.group("pde").lower()
    return pde, float(m.group("r"))


def bh(p_values) -> np.ndarray:
    p = np.asarray(p_values, float)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order]
    qrank = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    q = np.empty_like(qrank)
    q[order] = np.minimum(qrank, 1.0)
    out[finite] = q
    return out


def cliffs_delta(a, b) -> float:
    a = np.asarray(a, float)[:, None]
    b = np.asarray(b, float)[None, :]
    return float(((a > b).sum() - (a < b).sum()) / (a.size * b.size))


def load_seed_rows(trained_root: str, baseline_root: str) -> pd.DataFrame:
    rows = []
    condition_dirs = sorted(
        d for d in os.listdir(baseline_root)
        if os.path.isdir(os.path.join(baseline_root, d))
    )
    for condition in condition_dirs:
        pde, r = parse_condition(condition)
        trained_dir = os.path.join(trained_root, condition)
        baseline_dir = os.path.join(baseline_root, condition)
        if not os.path.isdir(trained_dir):
            raise FileNotFoundError(f"missing trained condition directory: {trained_dir}")

        seed_dirs = sorted(
            d for d in os.listdir(trained_dir)
            if re.fullmatch(r"seed_\d+", d)
        )
        trained_values = {}
        for seed_dir in seed_dirs:
            seed = int(seed_dir.split("_")[-1])
            rec = {"pde": pde, "r": r, "condition": condition,
                   "group": "trained", "seed": seed}
            for metric, (trained_name, _) in FILES.items():
                path = os.path.join(trained_dir, seed_dir, trained_name)
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
                matrix = np.asarray(np.load(path), float)
                if matrix.ndim != 2:
                    raise ValueError(f"{path}: expected (layer, head), got {matrix.shape}")
                rec[metric] = float(matrix.mean())
                if metric == "signed_corr":
                    rec["abs_corr"] = float(np.abs(matrix).mean())
            trained_values[seed] = rec
            rows.append(rec)

        arrays = {}
        for metric, (_, baseline_name) in FILES.items():
            path = os.path.join(baseline_dir, baseline_name)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            array = np.asarray(np.load(path), float)
            if array.ndim != 3:
                raise ValueError(f"{path}: expected (seed, layer, head), got {array.shape}")
            arrays[metric] = array
        counts = {a.shape[0] for a in arrays.values()}
        if len(counts) != 1:
            raise ValueError(f"baseline seed counts disagree for {condition}: {counts}")
        for seed in range(counts.pop()):
            rec = {"pde": pde, "r": r, "condition": condition,
                   "group": "random", "seed": seed}
            for metric, array in arrays.items():
                rec[metric] = float(array[seed].mean())
                if metric == "signed_corr":
                    rec["abs_corr"] = float(np.abs(array[seed]).mean())
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["pde", "r", "group", "seed"])


def compare(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["entropy", "signed_corr", "abs_corr", "locality"]
    for (pde, r, condition), group in seed_df.groupby(["pde", "r", "condition"]):
        trained = group[group.group == "trained"]
        random = group[group.group == "random"]
        for metric in metrics:
            a = trained[metric].to_numpy(float)
            b = random[metric].to_numpy(float)
            test = mannwhitneyu(a, b, alternative="two-sided", method="exact")
            rows.append({
                "pde": pde, "r": r, "condition": condition, "metric": metric,
                "n_trained": len(a), "n_random": len(b),
                "trained_mean": a.mean(), "trained_sd": a.std(ddof=1),
                "trained_median": np.median(a),
                "random_mean": b.mean(), "random_sd": b.std(ddof=1),
                "random_median": np.median(b),
                "mean_difference": a.mean() - b.mean(),
                "median_difference": np.median(a) - np.median(b),
                "cliffs_delta": cliffs_delta(a, b),
                "mannwhitney_u": float(test.statistic),
                "p_value": float(test.pvalue),
                "direction": "higher" if a.mean() > b.mean() else "lower",
            })
    out = pd.DataFrame(rows)
    out["p_value_bh"] = np.nan
    for metric, index in out.groupby("metric").groups.items():
        out.loc[index, "p_value_bh"] = bh(out.loc[index, "p_value"])
    out["significant_bh_0.05"] = out.p_value_bh < 0.05
    return out.sort_values(["metric", "pde", "r"])


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--trained-root", default="final_results")
    ap.add_argument("--baseline-root", default="final_results/random_baselines")
    ap.add_argument(
        "--out", default="analysis/what_do_heads_learn/results/trained_vs_random"
    )
    args = ap.parse_args()

    seed_df = load_seed_rows(args.trained_root, args.baseline_root)
    result = compare(seed_df)
    os.makedirs(args.out, exist_ok=True)
    seed_df.to_csv(os.path.join(args.out, "seed_metrics.csv"), index=False)
    result.to_csv(os.path.join(args.out, "trained_vs_random.csv"), index=False)

    show = ["metric", "pde", "r", "n_trained", "n_random", "trained_mean",
            "random_mean", "mean_difference", "cliffs_delta", "p_value",
            "p_value_bh", "significant_bh_0.05"]
    print(result[show].round(6).to_string(index=False))
    print("\nBH correction is separate across the nine conditions for each metric.")
    print("The independent unit is one model seed; heads are averaged within seed.")
    print(f"\nwrote {args.out}/seed_metrics.csv, trained_vs_random.csv")


if __name__ == "__main__":
    main()
