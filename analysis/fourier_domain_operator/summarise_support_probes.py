"""Seed-first, support-aware summaries of finite sinusoid probes."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from support_aware_analysis import support_region


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    probes = pd.read_csv(args.probes)
    nyquist = int(np.ceil(probes.normalized_frequency.max() * 32))
    # All current experiments use N=64; infer the exact Nyquist from
    # normalized_frequency = k/(N/2) rather than hard-coding the mode labels.
    inferred = probes[probes.normalized_frequency > 0]
    ratios = inferred["mode"] / inferred["normalized_frequency"]
    if len(ratios):
        nyquist = int(round(float(np.median(ratios))))
    probes["evaluation_scope"] = [
        support_region(int(mode), nyquist) for mode in probes["mode"]
    ]
    denom = probes.reference_distance + probes.identity_distance
    probes["reference_advantage"] = np.where(
        denom > 0,
        (probes.identity_distance - probes.reference_distance) / denom,
        0.0,
    )
    probes["closer_reference"] = probes.reference_distance < probes.identity_distance

    keys = [
        "pde", "r", "reference_scope", "seed", "amplitude_factor",
        "evaluation_scope",
    ]
    seed = probes.groupby(keys, as_index=False).agg(
        n_modes=("mode", "nunique"),
        rmse_reference=("reference_distance", lambda x: np.sqrt(np.mean(x * x))),
        rmse_identity=("identity_distance", lambda x: np.sqrt(np.mean(x * x))),
        reference_advantage=("reference_advantage", "mean"),
        fraction_modes_closer_reference=("closer_reference", "mean"),
        harmonic_power_fraction=("harmonic_power_fraction", "mean"),
        phase_equivariance_cv=("phase_equivariance_cv", "mean"),
    )

    group = ["pde", "r", "reference_scope", "amplitude_factor", "evaluation_scope"]
    metrics = [
        "rmse_reference", "rmse_identity", "reference_advantage",
        "fraction_modes_closer_reference", "harmonic_power_fraction",
        "phase_equivariance_cv",
    ]
    rows = []
    for values, frame in seed.groupby(group, sort=True):
        row = dict(zip(group, values))
        row["n_seeds"] = int(frame.seed.nunique())
        row["n_modes"] = int(frame.n_modes.iloc[0])
        for metric in metrics:
            x = frame[metric].dropna().to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(x)) if len(x) else np.nan
            row[f"{metric}_seed_p025"] = float(np.quantile(x, 0.025)) if len(x) else np.nan
            row[f"{metric}_seed_p975"] = float(np.quantile(x, 0.975)) if len(x) else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["pde", "r", "amplitude_factor", "evaluation_scope"]
    )

    os.makedirs(args.out, exist_ok=True)
    probes.to_csv(os.path.join(args.out, "sinusoid_probes_support_aware.csv"), index=False)
    seed.to_csv(os.path.join(args.out, "sinusoid_support_seed_metrics.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "sinusoid_support_summary.csv"), index=False)

    show = summary[summary.evaluation_scope == "in_support_k1_5"][[
        "pde", "r", "amplitude_factor", "n_modes", "rmse_reference_mean",
        "rmse_identity_mean", "reference_advantage_mean",
        "fraction_modes_closer_reference_mean", "harmonic_power_fraction_mean",
        "phase_equivariance_cv_mean",
    ]]
    print("in-support finite-amplitude probes (k=1..5)")
    print(show.round(5).to_string(index=False))
    print(f"\nwrote support-aware probe summaries to {args.out}")


if __name__ == "__main__":
    main()
