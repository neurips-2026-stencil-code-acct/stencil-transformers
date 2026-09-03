"""Recover circulant diagnostics for the learned update J-I.

The original Jacobian output stores the circular-diagonal profile P(J) and the
relative residual ||J-P(J)||_F / ||J||_F.  Here P is the Frobenius-orthogonal
projection onto circulant matrices.  Those saved quantities are sufficient to
recover the residual for J-I without loading a checkpoint.

Because identity is circulant and P is linear,

    (J-I) - P(J-I) = J - P(J).

The noncirculant numerator is therefore unchanged.  Orthogonality gives

    ||J||_F^2 = ||P(J)||_F^2 + ||J-P(J)||_F^2.

If r = ||J-P(J)||_F / ||J||_F, then

    ||J||_F^2 = ||P(J)||_F^2 / (1-r^2).

The saved profile determines P(J), including ||P(J)-I||_F.  Hence

    ||J-I||_F^2 = ||P(J)-I||_F^2 + ||J-P(J)||_F^2.

This derivation is exact apart from floating-point rounding.  It becomes
unidentifiable from the saved summary when P(J) is zero and r is one, and the
relative update residual is undefined when J-I has zero norm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_model_effects import bootstrap_mean, parse_pdes, require_columns


DEFAULT_JACOBIAN = Path("analysis/is_it_mechanistic/results/jacobian.csv")
DEFAULT_PROFILES = Path(
    "analysis/is_it_mechanistic/results/jacobian_profiles.npz"
)
DEFAULT_OUT = Path("analysis/robustness_checks/results/j_minus_identity")


def stored_profile_key(row: pd.Series, n_steps: int) -> str:
    stored_step = int(row["input_step"]) + (n_steps - 1)
    return (
        f"{row['pde']}_r{float(row['r']):g}_seed{int(row['seed'])}"
        f"_step{stored_step}"
    )


def reconstruct_update_diagnostics(
    profile: np.ndarray,
    offsets: np.ndarray,
    original_residual: float,
    tolerance: float = 1e-12,
) -> dict[str, float | str]:
    """Recover Frobenius diagnostics for J and J-I from saved Jacobian summaries."""
    profile = np.asarray(profile, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=int)
    if profile.ndim != 1 or offsets.shape != profile.shape:
        raise ValueError("profile and offsets must be one-dimensional and aligned")
    zero = np.flatnonzero(offsets == 0)
    if len(zero) != 1:
        raise ValueError("offsets must contain zero exactly once")
    if not np.isfinite(original_residual) or original_residual < 0.0:
        raise ValueError("original residual must be finite and non-negative")
    if original_residual > 1.0 + 1e-8:
        raise ValueError("a projection residual cannot exceed one")

    n = len(profile)
    residual = min(float(original_residual), 1.0)
    circulant_norm_sq = float(n * np.dot(profile, profile))
    one_minus_r2 = 1.0 - residual**2
    if one_minus_r2 <= tolerance:
        return {
            "reconstruction_status": "not_identifiable_from_saved_summary",
            "fro_norm_j": np.nan,
            "circulant_component_norm_j": float(np.sqrt(circulant_norm_sq)),
            "noncirculant_norm": np.nan,
            "fro_norm_j_minus_i": np.nan,
            "circulant_component_norm_j_minus_i": np.nan,
            "circulant_residual_j_minus_i": np.nan,
            "noncirculant_squared_norm_fraction_j_minus_i": np.nan,
            "update_norm_relative_to_j": np.nan,
        }

    norm_j_sq = circulant_norm_sq / one_minus_r2
    noncirculant_norm_sq = residual**2 * norm_j_sq
    update_profile = profile.copy()
    update_profile[zero[0]] -= 1.0
    update_circulant_norm_sq = float(n * np.dot(update_profile, update_profile))
    update_norm_sq = update_circulant_norm_sq + noncirculant_norm_sq

    if update_norm_sq <= tolerance:
        update_residual = np.nan
        update_fraction = np.nan
        status = "zero_update_norm"
    else:
        update_residual = float(np.sqrt(noncirculant_norm_sq / update_norm_sq))
        update_fraction = float(noncirculant_norm_sq / update_norm_sq)
        status = "ok"

    return {
        "reconstruction_status": status,
        "fro_norm_j": float(np.sqrt(norm_j_sq)),
        "circulant_component_norm_j": float(np.sqrt(circulant_norm_sq)),
        "noncirculant_norm": float(np.sqrt(noncirculant_norm_sq)),
        "fro_norm_j_minus_i": float(np.sqrt(max(update_norm_sq, 0.0))),
        "circulant_component_norm_j_minus_i": float(
            np.sqrt(update_circulant_norm_sq)
        ),
        "circulant_residual_j_minus_i": update_residual,
        "noncirculant_squared_norm_fraction_j_minus_i": update_fraction,
        "update_norm_relative_to_j": (
            float(np.sqrt(update_norm_sq / norm_j_sq)) if norm_j_sq > tolerance else np.nan
        ),
    }


def analyse_saved_results(
    jacobian: pd.DataFrame,
    profiles: np.lib.npyio.NpzFile,
    pdes: tuple[str, ...],
) -> pd.DataFrame:
    required = {"pde", "r", "seed", "input_step", "circulant_residual"}
    require_columns(jacobian, required, "Jacobian table")
    offsets = np.asarray(profiles["offsets"], dtype=int)
    jacobian = jacobian[jacobian["pde"].isin(pdes)].copy()
    if jacobian.empty:
        raise ValueError(f"no Jacobian rows found for PDEs {pdes}")
    if jacobian.duplicated(["pde", "r", "seed", "input_step"]).any():
        raise ValueError("Jacobian table has duplicate model/input-step rows")

    rows = []
    for _, row in jacobian.iterrows():
        same_model = jacobian[
            (jacobian["pde"] == row["pde"])
            & np.isclose(jacobian["r"], float(row["r"]))
            & (jacobian["seed"] == row["seed"])
        ]
        key = stored_profile_key(row, len(same_model))
        if key not in profiles.files:
            raise KeyError(f"saved Jacobian profile not found: {key}")
        diagnostics = reconstruct_update_diagnostics(
            profiles[key], offsets, float(row["circulant_residual"])
        )
        record = row.to_dict()
        record["profile_key"] = key
        record["n_grid"] = len(offsets)
        record["circulant_residual_j"] = float(row["circulant_residual"])
        record.update(diagnostics)
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["pde", "r", "seed", "input_step"])


def summarise_current_input(
    diagnostics: pd.DataFrame,
    n_bootstrap: int,
    confidence: float,
    master_seed: int,
) -> pd.DataFrame:
    current = diagnostics[
        (diagnostics["input_step"] == 0)
        & (diagnostics["reconstruction_status"] == "ok")
    ].copy()
    rows = []
    metrics = [
        "circulant_residual_j",
        "circulant_residual_j_minus_i",
        "noncirculant_squared_norm_fraction_j_minus_i",
        "update_norm_relative_to_j",
    ]
    for (pde, r), group in current.groupby(["pde", "r"], sort=True):
        row: dict[str, float | int | str] = {
            "pde": pde,
            "r": float(r),
            "n_model_seeds": int(group["seed"].nunique()),
            "confidence_level": confidence,
            "bootstrap_replicates": n_bootstrap,
            "bootstrap_seed": master_seed,
            "resampling_unit": "model seed within PDE-parameter condition",
        }
        for metric in metrics:
            estimate, lower, upper = bootstrap_mean(
                group[metric].to_numpy(),
                n_bootstrap,
                confidence,
                master_seed,
                f"jminus|{pde}|{float(r):g}|{metric}",
            )
            row[f"{metric}_mean"] = estimate
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pde", "r"])


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--jacobian", type=Path, default=DEFAULT_JACOBIAN)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pdes", default="heat,lf")
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20_260_824)
    args = parser.parse_args()

    pdes = parse_pdes(args.pdes)
    jacobian = pd.read_csv(args.jacobian)
    with np.load(args.profiles) as profiles:
        diagnostics = analyse_saved_results(jacobian, profiles, pdes)
    summary = summarise_current_input(
        diagnostics, args.n_bootstrap, args.confidence, args.seed
    )

    args.out.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(args.out / "j_minus_identity_seed_diagnostics.csv", index=False)
    summary.to_csv(args.out / "j_minus_identity_bootstrap_summary.csv", index=False)
    metadata = {
        "jacobian": str(args.jacobian),
        "profiles": str(args.profiles),
        "pdes": list(pdes),
        "n_bootstrap": args.n_bootstrap,
        "confidence": args.confidence,
        "bootstrap_seed": args.seed,
        "checkpoint_rerun_required": False,
        "reason": (
            "Identity is circulant, and circular-diagonal averaging is the "
            "Frobenius-orthogonal projection onto circulant matrices."
        ),
        "edge_cases": {
            "projection_zero_and_residual_one": (
                "absolute norms cannot be reconstructed from the saved relative residual"
            ),
            "j_equals_identity": "the relative residual of the zero update is undefined",
        },
    }
    (args.out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    show = summary[
        [
            "pde",
            "r",
            "circulant_residual_j_mean",
            "circulant_residual_j_minus_i_mean",
            "circulant_residual_j_minus_i_ci_lower",
            "circulant_residual_j_minus_i_ci_upper",
            "noncirculant_squared_norm_fraction_j_minus_i_mean",
        ]
    ]
    print("Circulant diagnostics for the full Jacobian J and learned update J-I")
    print(show.round(8).to_string(index=False))
    print(f"\nwrote derived outputs to {args.out}")


if __name__ == "__main__":
    main()
