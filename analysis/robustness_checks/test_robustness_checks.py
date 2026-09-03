"""Deterministic self-tests for the isolated robustness analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bootstrap_model_effects import (
    bootstrap_mean,
    head_preserving_seed_contrasts,
    summarise_head_preserving,
)
from j_minus_identity_circulance import reconstruct_update_diagnostics


def offsets_full(n: int) -> np.ndarray:
    return np.arange(-n // 2, n // 2)


def circulant_average(matrix: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    rows = np.arange(n)
    cols = (rows[:, None] + offsets[None, :]) % n
    return matrix[rows[:, None], cols].mean(axis=0)


def circulant_from_profile(profile: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    n = len(profile)
    first_row = np.zeros(n)
    first_row[offsets % n] = profile
    return np.stack([np.roll(first_row, row) for row in range(n)])


def relative_projection_residual(matrix: np.ndarray, offsets: np.ndarray) -> float:
    projection = circulant_from_profile(circulant_average(matrix, offsets), offsets)
    return float(np.linalg.norm(matrix - projection) / np.linalg.norm(matrix))


def main() -> None:
    n = 16
    offsets = offsets_full(n)
    rng = np.random.default_rng(9231)

    profile = rng.normal(size=n)
    circulant = circulant_from_profile(profile, offsets)
    raw = rng.normal(size=(n, n))
    noncirculant = raw - circulant_from_profile(
        circulant_average(raw, offsets), offsets
    )
    jacobian = circulant + 0.15 * noncirculant
    identity = np.eye(n)

    saved_profile = circulant_average(jacobian, offsets)
    residual_j = relative_projection_residual(jacobian, offsets)
    reconstructed = reconstruct_update_diagnostics(
        saved_profile, offsets, residual_j
    )

    projection_j = circulant_from_profile(saved_profile, offsets)
    update = jacobian - identity
    projection_update = circulant_from_profile(
        circulant_average(update, offsets), offsets
    )
    numerator_j = np.linalg.norm(jacobian - projection_j)
    numerator_update = np.linalg.norm(update - projection_update)
    direct_update_residual = numerator_update / np.linalg.norm(update)

    assert np.allclose(projection_update, projection_j - identity, atol=1e-12)
    assert abs(numerator_j - numerator_update) < 1e-12
    assert abs(
        reconstructed["circulant_residual_j_minus_i"] - direct_update_residual
    ) < 1e-12
    assert abs(reconstructed["noncirculant_norm"] - numerator_j) < 1e-12
    assert abs(reconstructed["fro_norm_j"] - np.linalg.norm(jacobian)) < 1e-12
    assert abs(reconstructed["fro_norm_j_minus_i"] - np.linalg.norm(update)) < 1e-12

    # A non-identity circulant matrix has zero residual before and after
    # subtracting identity.
    pure = reconstruct_update_diagnostics(profile, offsets, 0.0)
    assert pure["reconstruction_status"] == "ok"
    assert pure["circulant_residual_j_minus_i"] == 0.0

    # For J=I, J-I is the zero matrix, so a relative residual is undefined.
    identity_profile = np.zeros(n)
    identity_profile[offsets == 0] = 1.0
    zero_update = reconstruct_update_diagnostics(identity_profile, offsets, 0.0)
    assert zero_update["reconstruction_status"] == "zero_update_norm"
    assert np.isnan(zero_update["circulant_residual_j_minus_i"])

    # If the saved projection is zero and the original relative residual is
    # one, the original matrix norm was not saved and cannot be reconstructed.
    unidentified = reconstruct_update_diagnostics(np.zeros(n), offsets, 1.0)
    assert unidentified["reconstruction_status"] == (
        "not_identifiable_from_saved_summary"
    )

    constant = np.full(20, np.log(3.0))
    first = bootstrap_mean(constant, 1_000, 0.95, 123, "constant")
    second = bootstrap_mean(constant, 1_000, 0.95, 123, "constant")
    assert first == second
    assert np.allclose(first, np.log(3.0))

    substitution_values = {
        "original": (1.0, 0.0, "none"),
        "frozen_per_head_validation": (2.0, 0.1, "validation"),
        "frozen_layer_mean_validation": (4.0, 0.3, "validation"),
        "analytical_stencil": (8.0, 1.1, "validation"),
    }
    substitution_rows = []
    for seed in range(3):
        for substitution, (mse, excess, calibration_split) in (
            substitution_values.items()
        ):
            substitution_rows.append(
                {
                    "pde": "heat",
                    "r": 0.1,
                    "seed": seed,
                    "target": "all",
                    "substitution": substitution,
                    "mse": mse,
                    "excess": excess,
                    "calibration_split": calibration_split,
                    "evaluation_split": "test",
                }
            )
    head_effects = head_preserving_seed_contrasts(
        pd.DataFrame(substitution_rows), ("heat",)
    )
    head_summary = summarise_head_preserving(
        head_effects, 1_000, 0.95, 123
    )
    assert len(head_effects) == 9
    assert len(head_summary) == 3
    assert (head_summary["n_mse_higher"] == 3).all()
    assert (head_summary["n_mse_lower"] == 0).all()
    assert (head_summary["n_mse_equal"] == 0).all()
    ratios = dict(
        zip(head_summary["contrast"], head_summary["geometric_mean_mse_ratio"])
    )
    assert np.isclose(ratios["frozen_per_head_validation_vs_original"], 2.0)
    assert np.isclose(
        ratios[
            "frozen_layer_mean_validation_vs_frozen_per_head_validation"
        ],
        2.0,
    )
    assert np.isclose(
        ratios["analytical_stencil_vs_frozen_per_head_validation"], 4.0
    )

    print("robustness self-tests passed")
    print(f"noncirculant numerator difference: {abs(numerator_j - numerator_update):.2e}")
    print(
        "J-I residual reconstruction error: "
        f"{abs(reconstructed['circulant_residual_j_minus_i'] - direct_update_residual):.2e}"
    )
    print("bootstrap determinism and constant-sample interval: passed")
    print("head-preserving paired contrasts and bootstrap summary: passed")


if __name__ == "__main__":
    main()
