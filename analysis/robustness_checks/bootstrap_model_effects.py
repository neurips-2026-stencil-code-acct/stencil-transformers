"""Model-seed bootstrap intervals for substitutions and Fourier errors.

This script reads existing result tables and writes derived tables only.  It
does not load checkpoints, evaluate models, or modify the source results.

For every PDE--parameter condition, a model seed is the resampling unit.  A
seed's two measurements are differenced before resampling, so both members of
each comparison always remain paired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SUBSTITUTIONS = Path(
    "analysis/is_it_mechanistic/results/substitution.csv"
)
DEFAULT_HEAD_PRESERVING_SUBSTITUTIONS = Path(
    "analysis/is_it_mechanistic/results/head_preserving/substitution.csv"
)
DEFAULT_FOURIER = Path(
    "analysis/fourier_domain_operator/results/fourier_support_metrics.csv"
)
DEFAULT_OUT = Path("analysis/robustness_checks/results/paired_bootstrap")

HEAD_PRESERVING_CONTRASTS = (
    ("frozen_per_head_validation", "original"),
    ("frozen_layer_mean_validation", "frozen_per_head_validation"),
    ("analytical_stencil", "frozen_per_head_validation"),
)

FOURIER_SCOPES = (
    ("k1_5_uniform", "in_support_k1_5", "uniform"),
    (
        "k1_5_empirical_power",
        "empirical_training_spectrum_k1_5",
        "empirical_training_power",
    ),
    ("k6_8_uniform", "near_support_k6_8", "uniform"),
    (
        "k9_32_uniform",
        "spectral_extrapolation_k9_nyquist",
        "uniform",
    ),
)


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def parse_pdes(value: str) -> tuple[str, ...]:
    pdes = tuple(part.strip() for part in value.split(",") if part.strip())
    if not pdes:
        raise ValueError("--pdes must contain at least one PDE name")
    return pdes


def group_rng(master_seed: int, label: str) -> np.random.Generator:
    """Return an order-independent deterministic generator for one group."""
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    label_seed = int.from_bytes(digest[:8], "little", signed=False)
    sequence = np.random.SeedSequence(
        [master_seed, label_seed & 0xFFFFFFFF, label_seed >> 32]
    )
    return np.random.default_rng(sequence)


def bootstrap_mean(
    values: np.ndarray,
    n_bootstrap: int,
    confidence: float,
    master_seed: int,
    label: str,
) -> tuple[float, float, float]:
    """Percentile interval for a mean, resampling complete model seeds."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("bootstrap_mean requires at least two model-seed values")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite value in bootstrap group {label}")
    if n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("--confidence must lie between zero and one")

    rng = group_rng(master_seed, label)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return float(values.mean()), float(lower), float(upper)


def all_layer_seed_contrasts(frame: pd.DataFrame, pdes: tuple[str, ...]) -> pd.DataFrame:
    """Create one paired all-layer substitution contrast per model seed."""
    required = {
        "pde",
        "r",
        "seed",
        "target",
        "substitution",
        "mse",
        "mse_original",
    }
    require_columns(frame, required, "all-layer substitution table")
    frame = frame[(frame["target"] == "all") & frame["pde"].isin(pdes)].copy()
    if frame.empty:
        raise ValueError(f"no all-layer substitution rows found for PDEs {pdes}")
    keys = ["pde", "r", "seed"]
    if frame.duplicated(keys + ["substitution"]).any():
        raise ValueError("substitution table has duplicate seed/substitution rows")
    if (frame["mse"] <= 0).any() or (frame["mse_original"] <= 0).any():
        raise ValueError("substitution MSE values must be positive for log ratios")

    pivot = frame.pivot(index=keys, columns="substitution", values="mse")
    if "original" not in pivot:
        raise ValueError("substitution table has no original-attention rows")
    original_check = frame.pivot(
        index=keys, columns="substitution", values="mse_original"
    )
    if not np.allclose(
        pivot["original"].to_numpy(),
        original_check.to_numpy().ravel()[:: original_check.shape[1]],
        rtol=1e-10,
        atol=0.0,
    ):
        # The compact check above depends on column order.  Fall back to an
        # explicit row check before declaring a mismatch.
        joined = frame.merge(
            pivot["original"].rename("original_row_mse").reset_index(), on=keys
        )
        if not np.allclose(
            joined["mse_original"], joined["original_row_mse"], rtol=1e-10, atol=0.0
        ):
            raise ValueError("mse_original disagrees with the original row")

    contrasts: list[tuple[str, str]] = [
        (name, "original") for name in sorted(pivot.columns) if name != "original"
    ]
    if {"stencil", "frozen"}.issubset(pivot.columns):
        contrasts.append(("stencil", "frozen"))

    rows: list[dict[str, float | int | str]] = []
    base = pivot.reset_index()
    for comparison, baseline in contrasts:
        selected = base[keys + [comparison, baseline]].dropna()
        for item in selected.itertuples(index=False, name=None):
            pde, r, seed, mse_comparison, mse_baseline = item
            ratio = float(mse_comparison / mse_baseline)
            rows.append(
                {
                    "pde": pde,
                    "r": float(r),
                    "seed": int(seed),
                    "comparison": comparison,
                    "baseline": baseline,
                    "contrast": f"{comparison}_vs_{baseline}",
                    "mse_comparison": float(mse_comparison),
                    "mse_baseline": float(mse_baseline),
                    "mse_ratio": ratio,
                    "log_mse_ratio": float(np.log(ratio)),
                    "percent_mse_change": 100.0 * (ratio - 1.0),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["pde", "r", "comparison", "baseline", "seed"]
    )


def summarise_all_layer(
    seed_effects: pd.DataFrame,
    n_bootstrap: int,
    confidence: float,
    master_seed: int,
) -> pd.DataFrame:
    rows = []
    keys = ["pde", "r", "comparison", "baseline", "contrast"]
    for values, group in seed_effects.groupby(keys, sort=True):
        # Keep this legacy RNG namespace so renaming code cannot change the
        # already reported deterministic bootstrap intervals.
        label = "q4|" + "|".join(str(value) for value in values)
        estimate, lower, upper = bootstrap_mean(
            group["log_mse_ratio"].to_numpy(),
            n_bootstrap,
            confidence,
            master_seed,
            label,
        )
        ratio, ratio_lower, ratio_upper = np.exp([estimate, lower, upper])
        row = dict(zip(keys, values))
        row.update(
            {
                "n_model_seeds": int(group["seed"].nunique()),
                "mean_log_mse_ratio": estimate,
                "mean_log_mse_ratio_ci_lower": lower,
                "mean_log_mse_ratio_ci_upper": upper,
                "geometric_mean_mse_ratio": float(ratio),
                "geometric_mean_mse_ratio_ci_lower": float(ratio_lower),
                "geometric_mean_mse_ratio_ci_upper": float(ratio_upper),
                "percent_mse_change": float(100.0 * (ratio - 1.0)),
                "percent_mse_change_ci_lower": float(100.0 * (ratio_lower - 1.0)),
                "percent_mse_change_ci_upper": float(100.0 * (ratio_upper - 1.0)),
                "fraction_model_seeds_with_higher_mse": float(
                    np.mean(group["log_mse_ratio"] > 0.0)
                ),
                "confidence_level": confidence,
                "bootstrap_replicates": n_bootstrap,
                "bootstrap_seed": master_seed,
                "resampling_unit": "model seed within PDE-parameter condition",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys)


def head_preserving_seed_contrasts(
    frame: pd.DataFrame,
    pdes: tuple[str, ...],
) -> pd.DataFrame:
    """Create paired contrasts for the validation-derived attention controls."""
    required = {
        "pde",
        "r",
        "seed",
        "target",
        "substitution",
        "mse",
        "excess",
        "calibration_split",
        "evaluation_split",
    }
    require_columns(frame, required, "head-preserving substitution table")
    frame = frame[(frame["target"] == "all") & frame["pde"].isin(pdes)].copy()
    if frame.empty:
        raise ValueError(f"no head-preserving substitution rows found for PDEs {pdes}")

    keys = ["pde", "r", "seed"]
    if frame.duplicated(keys + ["substitution"]).any():
        raise ValueError(
            "head-preserving table has duplicate seed/substitution rows"
        )
    if (frame["mse"] <= 0).any():
        raise ValueError("head-preserving MSE values must be positive")
    if not np.isfinite(frame[["mse", "excess"]].to_numpy()).all():
        raise ValueError("head-preserving effects must be finite")

    required_substitutions = {
        "original",
        *[name for pair in HEAD_PRESERVING_CONTRASTS for name in pair],
    }
    observed = set(frame["substitution"])
    missing = required_substitutions - observed
    if missing:
        raise ValueError(
            "head-preserving table is missing substitutions: "
            f"{sorted(missing)}"
        )
    changed = frame[frame["substitution"] != "original"]
    if not (changed["calibration_split"] == "validation").all():
        raise ValueError("every fixed attention matrix must come from validation data")
    if not (frame["evaluation_split"] == "test").all():
        raise ValueError("every head-preserving row must be evaluated on test data")

    rows = []
    for comparison, baseline in HEAD_PRESERVING_CONTRASTS:
        comparison_rows = frame[frame["substitution"] == comparison][
            keys + ["mse", "excess"]
        ].rename(columns={"mse": "mse_comparison", "excess": "excess_comparison"})
        baseline_rows = frame[frame["substitution"] == baseline][
            keys + ["mse", "excess"]
        ].rename(columns={"mse": "mse_baseline", "excess": "excess_baseline"})
        paired = comparison_rows.merge(
            baseline_rows,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(comparison_rows) or len(paired) != len(baseline_rows):
            raise ValueError(f"unpaired model rows for {comparison} versus {baseline}")
        paired["comparison"] = comparison
        paired["baseline"] = baseline
        paired["contrast"] = f"{comparison}_vs_{baseline}"
        paired["mse_difference"] = (
            paired["mse_comparison"] - paired["mse_baseline"]
        )
        paired["mse_ratio"] = paired["mse_comparison"] / paired["mse_baseline"]
        paired["log_mse_ratio"] = np.log(paired["mse_ratio"])
        paired["excess_difference"] = (
            paired["excess_comparison"] - paired["excess_baseline"]
        )
        paired["mse_sign"] = np.where(
            paired["mse_difference"] > 0.0,
            "higher",
            np.where(paired["mse_difference"] < 0.0, "lower", "equal"),
        )
        rows.append(paired)

    columns = [
        "pde",
        "r",
        "seed",
        "comparison",
        "baseline",
        "contrast",
        "mse_comparison",
        "mse_baseline",
        "mse_difference",
        "mse_ratio",
        "log_mse_ratio",
        "excess_comparison",
        "excess_baseline",
        "excess_difference",
        "mse_sign",
    ]
    return pd.concat(rows, ignore_index=True)[columns].sort_values(
        ["pde", "r", "comparison", "baseline", "seed"]
    )


def summarise_head_preserving(
    seed_effects: pd.DataFrame,
    n_bootstrap: int,
    confidence: float,
    master_seed: int,
) -> pd.DataFrame:
    """Bootstrap paired head-preserving effects over complete model seeds."""
    rows = []
    keys = ["pde", "r", "comparison", "baseline", "contrast"]
    for values, group in seed_effects.groupby(keys, sort=True):
        # Keep this legacy RNG namespace so renaming code cannot change the
        # already reported deterministic bootstrap intervals.
        label_base = "q4_head_preserving|" + "|".join(
            str(value) for value in values
        )
        log_estimate, log_lower, log_upper = bootstrap_mean(
            group["log_mse_ratio"].to_numpy(),
            n_bootstrap,
            confidence,
            master_seed,
            label_base + "|log_mse_ratio",
        )
        raw_estimate, raw_lower, raw_upper = bootstrap_mean(
            group["mse_difference"].to_numpy(),
            n_bootstrap,
            confidence,
            master_seed,
            label_base + "|raw_mse_difference",
        )
        excess_estimate, excess_lower, excess_upper = bootstrap_mean(
            group["excess_difference"].to_numpy(),
            n_bootstrap,
            confidence,
            master_seed,
            label_base + "|excess_difference",
        )
        ratio, ratio_lower, ratio_upper = np.exp(
            [log_estimate, log_lower, log_upper]
        )
        row = dict(zip(keys, values))
        row.update(
            {
                "n_model_seeds": int(group["seed"].nunique()),
                "n_mse_higher": int((group["mse_sign"] == "higher").sum()),
                "n_mse_lower": int((group["mse_sign"] == "lower").sum()),
                "n_mse_equal": int((group["mse_sign"] == "equal").sum()),
                "mean_log_mse_ratio": log_estimate,
                "mean_log_mse_ratio_ci_lower": log_lower,
                "mean_log_mse_ratio_ci_upper": log_upper,
                "geometric_mean_mse_ratio": float(ratio),
                "geometric_mean_mse_ratio_ci_lower": float(ratio_lower),
                "geometric_mean_mse_ratio_ci_upper": float(ratio_upper),
                "mean_raw_mse_difference": raw_estimate,
                "mean_raw_mse_difference_ci_lower": raw_lower,
                "mean_raw_mse_difference_ci_upper": raw_upper,
                "mean_excess_difference": excess_estimate,
                "mean_excess_difference_ci_lower": excess_lower,
                "mean_excess_difference_ci_upper": excess_upper,
                "mean_excess_comparison": float(group["excess_comparison"].mean()),
                "mean_excess_baseline": float(group["excess_baseline"].mean()),
                "confidence_level": confidence,
                "bootstrap_replicates": n_bootstrap,
                "bootstrap_seed": master_seed,
                "resampling_unit": "model seed within PDE-parameter condition",
                "calibration_split": "validation",
                "evaluation_split": "test",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys)


def fourier_seed_deltas(frame: pd.DataFrame, pdes: tuple[str, ...]) -> pd.DataFrame:
    """Create seed-level raw identity-minus-PDE-reference RMSE differences."""
    required = {
        "pde",
        "r",
        "seed",
        "input_step_index",
        "reference_scope",
        "evaluation_scope",
        "weighting",
        "rmse_reference",
        "rmse_identity",
    }
    require_columns(frame, required, "support-aware Fourier table")
    frame = frame[(frame["input_step_index"] == 0) & frame["pde"].isin(pdes)].copy()
    if frame.empty:
        raise ValueError(f"no current-input Fourier rows found for PDEs {pdes}")

    rows = []
    for scope_label, scope, weighting in FOURIER_SCOPES:
        selected = frame[
            (frame["evaluation_scope"] == scope) & (frame["weighting"] == weighting)
        ].copy()
        selected["scope_label"] = scope_label
        rows.append(selected)
    selected = pd.concat(rows, ignore_index=True)
    keys = ["pde", "r", "seed", "reference_scope", "scope_label"]
    if selected.duplicated(keys).any():
        raise ValueError("Fourier table has duplicate model-seed/scope rows")
    if selected[["rmse_reference", "rmse_identity"]].isna().any().any():
        raise ValueError("Fourier RMSE values must be finite")
    selected["delta_identity_minus_pde"] = (
        selected["rmse_identity"] - selected["rmse_reference"]
    )
    selected["closer_reference"] = np.where(
        selected["delta_identity_minus_pde"] > 0.0,
        "PDE reference",
        np.where(
            selected["delta_identity_minus_pde"] < 0.0, "identity", "equal"
        ),
    )
    columns = [
        "pde",
        "r",
        "seed",
        "reference_scope",
        "scope_label",
        "evaluation_scope",
        "weighting",
        "n_modes",
        "mode_min",
        "mode_max",
        "rmse_reference",
        "rmse_identity",
        "delta_identity_minus_pde",
        "closer_reference",
    ]
    return selected[columns].sort_values(["pde", "r", "scope_label", "seed"])


def summarise_fourier(
    seed_effects: pd.DataFrame,
    n_bootstrap: int,
    confidence: float,
    master_seed: int,
) -> pd.DataFrame:
    rows = []
    keys = [
        "pde",
        "r",
        "reference_scope",
        "scope_label",
        "evaluation_scope",
        "weighting",
    ]
    for values, group in seed_effects.groupby(keys, sort=True):
        # Keep this legacy RNG namespace so renaming code cannot change the
        # already reported deterministic bootstrap intervals.
        label = "q5|" + "|".join(str(value) for value in values)
        estimate, lower, upper = bootstrap_mean(
            group["delta_identity_minus_pde"].to_numpy(),
            n_bootstrap,
            confidence,
            master_seed,
            label,
        )
        row = dict(zip(keys, values))
        row.update(
            {
                "n_model_seeds": int(group["seed"].nunique()),
                "n_modes": int(group["n_modes"].iloc[0]),
                "mode_min": int(group["mode_min"].iloc[0]),
                "mode_max": int(group["mode_max"].iloc[0]),
                "mean_rmse_pde_reference": float(group["rmse_reference"].mean()),
                "mean_rmse_identity": float(group["rmse_identity"].mean()),
                "mean_delta_identity_minus_pde": estimate,
                "mean_delta_ci_lower": lower,
                "mean_delta_ci_upper": upper,
                "fraction_model_seeds_closer_to_pde": float(
                    np.mean(group["delta_identity_minus_pde"] > 0.0)
                ),
                "confidence_level": confidence,
                "bootstrap_replicates": n_bootstrap,
                "bootstrap_seed": master_seed,
                "resampling_unit": "model seed within PDE-parameter condition",
                "interpretation": (
                    "positive delta means the PDE reference is closer; "
                    "negative delta means identity is closer"
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--substitutions", type=Path, default=DEFAULT_SUBSTITUTIONS)
    parser.add_argument(
        "--head-preserving-substitutions",
        type=Path,
        default=DEFAULT_HEAD_PRESERVING_SUBSTITUTIONS,
    )
    parser.add_argument("--fourier-metrics", type=Path, default=DEFAULT_FOURIER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--pdes",
        default="heat,lf",
        help="comma-separated PDE names; defaults to the workshop scope",
    )
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20_260_824)
    args = parser.parse_args()

    pdes = parse_pdes(args.pdes)
    substitutions_input = pd.read_csv(args.substitutions)
    head_preserving_input = pd.read_csv(args.head_preserving_substitutions)
    fourier_input = pd.read_csv(args.fourier_metrics)

    all_layer_seeds = all_layer_seed_contrasts(substitutions_input, pdes)
    all_layer_summary = summarise_all_layer(
        all_layer_seeds, args.n_bootstrap, args.confidence, args.seed
    )
    head_preserving_seeds = head_preserving_seed_contrasts(
        head_preserving_input, pdes
    )
    head_preserving_summary = summarise_head_preserving(
        head_preserving_seeds,
        args.n_bootstrap,
        args.confidence,
        args.seed,
    )
    fourier_seeds = fourier_seed_deltas(fourier_input, pdes)
    fourier_summary = summarise_fourier(
        fourier_seeds, args.n_bootstrap, args.confidence, args.seed
    )

    args.out.mkdir(parents=True, exist_ok=True)
    all_layer_seeds.to_csv(args.out / "all_layer_seed_contrasts.csv", index=False)
    all_layer_summary.to_csv(args.out / "all_layer_bootstrap_summary.csv", index=False)
    head_preserving_seeds.to_csv(
        args.out / "head_preserving_seed_contrasts.csv", index=False
    )
    head_preserving_summary.to_csv(
        args.out / "head_preserving_bootstrap_summary.csv", index=False
    )
    fourier_seeds.to_csv(args.out / "fourier_seed_deltas.csv", index=False)
    fourier_summary.to_csv(args.out / "fourier_bootstrap_summary.csv", index=False)
    metadata = {
        "substitutions": str(args.substitutions),
        "head_preserving_substitutions": str(args.head_preserving_substitutions),
        "fourier_metrics": str(args.fourier_metrics),
        "pdes": list(pdes),
        "n_bootstrap": args.n_bootstrap,
        "confidence": args.confidence,
        "bootstrap_seed": args.seed,
        "resampling": (
            "Within each PDE-parameter condition, sample model seeds with "
            "replacement. Compute each paired effect before resampling."
        ),
    }
    (args.out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    head_preserving_metadata = {
        "source": str(args.head_preserving_substitutions),
        "pdes": list(pdes),
        "contrasts": [list(pair) for pair in HEAD_PRESERVING_CONTRASTS],
        "n_bootstrap": args.n_bootstrap,
        "confidence": args.confidence,
        "bootstrap_seed": args.seed,
        "resampling": (
            "Paired effects are computed within each model, then model seeds "
            "are sampled with replacement within each PDE-parameter condition."
        ),
        "primary_effect": "mean paired log MSE ratio",
        "separate_secondary_effect": "mean paired excess-error difference",
    }
    (args.out / "head_preserving_bootstrap_metadata.json").write_text(
        json.dumps(head_preserving_metadata, indent=2) + "\n", encoding="utf-8"
    )

    all_layer_key = all_layer_summary[
        all_layer_summary["contrast"].isin(
            ["frozen_vs_original", "stencil_vs_original", "stencil_vs_frozen"]
        )
    ][
        [
            "pde",
            "r",
            "contrast",
            "geometric_mean_mse_ratio",
            "geometric_mean_mse_ratio_ci_lower",
            "geometric_mean_mse_ratio_ci_upper",
        ]
    ]
    fourier_show = fourier_summary[
        [
            "pde",
            "r",
            "scope_label",
            "mean_delta_identity_minus_pde",
            "mean_delta_ci_lower",
            "mean_delta_ci_upper",
        ]
    ]
    print("Paired all-layer effects (geometric mean MSE ratio)")
    print(all_layer_key.round(6).to_string(index=False))
    head_preserving_show = head_preserving_summary[
        [
            "pde",
            "r",
            "contrast",
            "geometric_mean_mse_ratio",
            "geometric_mean_mse_ratio_ci_lower",
            "geometric_mean_mse_ratio_ci_upper",
            "mean_excess_difference",
            "mean_excess_difference_ci_lower",
            "mean_excess_difference_ci_upper",
        ]
    ]
    print("\nValidation-derived head-preserving effects")
    print(head_preserving_show.round(6).to_string(index=False))
    print("\nFourier paired raw delta = RMSE_identity - RMSE_PDE")
    print(fourier_show.round(6).to_string(index=False))
    print(f"\nwrote derived outputs to {args.out}")


if __name__ == "__main__":
    main()
