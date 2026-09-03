"""Recompute false-discovery-rate correction for the workshop scope.

The original trained-versus-random table corrects each metric across nine
PDE/parameter conditions.  The workshop paper uses only heat and
Lax--Friedrichs at parameters 0.1, 0.25, and 0.4.  This script preserves the
original tests and effect sizes, selects those six conditions, and recomputes
Benjamini--Hochberg adjusted p-values separately for each metric.

No model is trained or evaluated by this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WORKSHOP_PDES = {"heat", "lf"}
WORKSHOP_PARAMETERS = {0.1, 0.25, 0.4}


def canonical_pde(value: str) -> str:
    name = str(value).strip().lower()
    if name in {"lf", "lax_friedrichs", "lax-friedrichs", "advection"}:
        return "lf"
    return name


def bh_adjust(p_values) -> np.ndarray:
    """Benjamini--Hochberg adjusted p-values, preserving missing entries."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    values = p[finite]
    if not len(values):
        return adjusted
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie between zero and one")

    order = np.argsort(values)
    ranked = values[order]
    monotone = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(monotone)
    restored[order] = np.minimum(monotone, 1.0)
    adjusted[finite] = restored
    return adjusted


def select_workshop_conditions(table: pd.DataFrame) -> pd.DataFrame:
    required = {"pde", "r", "metric", "p_value"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"input is missing columns: {', '.join(missing)}")

    out = table.copy()
    out["pde"] = out["pde"].map(canonical_pde)
    out["r"] = pd.to_numeric(out["r"], errors="raise")
    parameter_key = out["r"].round(8)
    keep = out["pde"].isin(WORKSHOP_PDES) & parameter_key.isin(WORKSHOP_PARAMETERS)
    out = out.loc[keep].copy()

    expected = {(pde, r) for pde in WORKSHOP_PDES for r in WORKSHOP_PARAMETERS}
    for metric, group in out.groupby("metric", sort=True):
        observed = set(zip(group["pde"], group["r"].round(8)))
        if observed != expected:
            missing_conditions = sorted(expected - observed)
            extra_conditions = sorted(observed - expected)
            raise ValueError(
                f"metric {metric!r} does not contain exactly the six workshop "
                f"conditions; missing={missing_conditions}, extra={extra_conditions}"
            )
        if len(group) != 6:
            raise ValueError(
                f"metric {metric!r} has {len(group)} rows; expected one row for "
                "each of six workshop conditions"
            )
    if out.empty:
        raise ValueError("no workshop conditions were found")
    return out


def recompute_workshop_bh(table: pd.DataFrame) -> pd.DataFrame:
    out = select_workshop_conditions(table)
    rename = {}
    if "p_value_bh" in out.columns:
        rename["p_value_bh"] = "p_value_bh_original_nine_conditions"
    if "significant_bh_0.05" in out.columns:
        rename["significant_bh_0.05"] = (
            "significant_bh_original_nine_conditions_0.05"
        )
    out = out.rename(columns=rename)

    out["p_value_bh_six_conditions"] = np.nan
    for _, index in out.groupby("metric", sort=True).groups.items():
        out.loc[index, "p_value_bh_six_conditions"] = bh_adjust(
            out.loc[index, "p_value"].to_numpy(float)
        )
    out["significant_bh_six_conditions_0.05"] = (
        out["p_value_bh_six_conditions"] < 0.05
    )
    out["bh_family_size"] = 6
    return out.sort_values(["metric", "pde", "r"]).reset_index(drop=True)


def build_summary(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, group in result.groupby("metric", sort=True):
        row = {
            "metric": metric,
            "bh_family_size": 6,
            "n_significant_six_conditions_0.05": int(
                group["significant_bh_six_conditions_0.05"].sum()
            ),
            "minimum_adjusted_p_six_conditions": float(
                group["p_value_bh_six_conditions"].min()
            ),
            "maximum_adjusted_p_six_conditions": float(
                group["p_value_bh_six_conditions"].max()
            ),
        }
        old = "significant_bh_original_nine_conditions_0.05"
        if old in group:
            row["n_significant_original_nine_conditions_0.05"] = int(
                group[old].astype(bool).sum()
            )
            row["n_decisions_changed_at_0.05"] = int(
                (group[old].astype(bool)
                 != group["significant_bh_six_conditions_0.05"]).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--input",
        default="analysis/what_do_heads_learn/results/trained_vs_random/trained_vs_random.csv",
    )
    parser.add_argument(
        "--out",
        default="analysis/what_do_heads_learn/results/robustness",
    )
    args = parser.parse_args()

    result = recompute_workshop_bh(pd.read_csv(args.input))
    summary = build_summary(result)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "trained_vs_random_bh_six_conditions.csv", index=False)
    summary.to_csv(out_dir / "trained_vs_random_bh_summary.csv", index=False)

    columns = [
        "metric", "pde", "r", "p_value", "p_value_bh_six_conditions",
        "significant_bh_six_conditions_0.05",
    ]
    print(result[columns].to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nWrote workshop-scope BH results to {out_dir}")


if __name__ == "__main__":
    main()
