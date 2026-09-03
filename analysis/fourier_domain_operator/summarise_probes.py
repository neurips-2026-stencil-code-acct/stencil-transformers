"""Create seed-first summaries from finite-amplitude probe rows."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = pd.read_csv(args.probes)
    denom = p.reference_distance + p.identity_distance
    p["reference_advantage"] = np.where(
        denom > 0,
        (p.identity_distance - p.reference_distance) / denom,
        0.0,
    )
    p["closer_reference"] = p.reference_distance < p.identity_distance
    keys = ["pde", "r", "reference_scope", "seed", "amplitude_factor"]
    seed = p.groupby(keys, as_index=False).agg(
        rmse_reference=("reference_distance", lambda x: np.sqrt(np.mean(x * x))),
        rmse_identity=("identity_distance", lambda x: np.sqrt(np.mean(x * x))),
        reference_advantage=("reference_advantage", "mean"),
        fraction_modes_closer_reference=("closer_reference", "mean"),
        harmonic_power_fraction=("harmonic_power_fraction", "mean"),
        phase_equivariance_cv=("phase_equivariance_cv", "mean"),
    )

    group = ["pde", "r", "reference_scope", "amplitude_factor"]
    metrics = [c for c in seed.columns if c not in keys]
    records = []
    for values, frame in seed.groupby(group):
        row = dict(zip(group, values))
        row["n_seeds"] = len(frame)
        for metric in metrics:
            x = frame[metric].to_numpy(float)
            row[f"{metric}_mean"] = np.mean(x)
            row[f"{metric}_seed_p025"] = np.quantile(x, 0.025)
            row[f"{metric}_seed_p975"] = np.quantile(x, 0.975)
        records.append(row)
    summary = pd.DataFrame(records).sort_values(["pde", "r", "amplitude_factor"])

    amplitudes = sorted(p.amplitude_factor.unique())
    sensitivity = pd.DataFrame()
    if len(amplitudes) >= 2:
        lo, hi = amplitudes[0], amplitudes[-1]
        index = ["pde", "r", "reference_scope", "seed", "mode"]
        real = p.pivot(index=index, columns="amplitude_factor", values="response_real")
        imag = p.pivot(index=index, columns="amplitude_factor", values="response_imag")
        wide = real + 1j * imag
        wide = wide.dropna(subset=[lo, hi])
        change = np.abs(wide[hi] - wide[lo]) / (
            0.5 * (np.abs(wide[hi]) + np.abs(wide[lo])) + 1e-12
        )
        per_seed = change.rename("relative_symbol_change").reset_index().groupby(
            ["pde", "r", "reference_scope", "seed"], as_index=False
        ).relative_symbol_change.mean()
        sensitivity = per_seed.groupby(
            ["pde", "r", "reference_scope"], as_index=False
        ).agg(
            n_seeds=("seed", "size"),
            relative_symbol_change_mean=("relative_symbol_change", "mean"),
            relative_symbol_change_seed_p025=(
                "relative_symbol_change", lambda x: np.quantile(x, 0.025)),
            relative_symbol_change_seed_p975=(
                "relative_symbol_change", lambda x: np.quantile(x, 0.975)),
        )
        sensitivity["amplitude_factor_low"] = lo
        sensitivity["amplitude_factor_high"] = hi

    os.makedirs(args.out, exist_ok=True)
    seed.to_csv(os.path.join(args.out, "sinusoid_seed_metrics.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "sinusoid_summary.csv"), index=False)
    sensitivity.to_csv(
        os.path.join(args.out, "sinusoid_amplitude_sensitivity.csv"), index=False
    )

    show = ["pde", "r", "amplitude_factor", "rmse_reference_mean",
            "rmse_identity_mean", "reference_advantage_mean",
            "fraction_modes_closer_reference_mean",
            "harmonic_power_fraction_mean", "phase_equivariance_cv_mean"]
    print(summary[show].round(4).to_string(index=False))
    print("\namplitude sensitivity")
    print(sensitivity[["pde", "r", "relative_symbol_change_mean",
                       "relative_symbol_change_seed_p025",
                       "relative_symbol_change_seed_p975"]].round(4).to_string(index=False))
    print(f"\nwrote probe summaries to {args.out}")


if __name__ == "__main__":
    main()
