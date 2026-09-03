"""Exact tests for Fourier direction, support scopes, and weighted errors."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fourier_direction import BASE_ANALYSIS
from support_aware_analysis import (
    evaluation_scopes,
    selected_symbol_metrics,
    support_region,
)


def main() -> None:
    n = 64
    offsets = np.array([-1, 0, 1])
    modes = np.arange(n // 2 + 1)
    theta = 2 * np.pi * modes / n

    heat = BASE_ANALYSIS.symbol_from_offsets(
        np.array([0.25, 0.5, 0.25]), offsets, n
    )
    heat_expected = 0.5 + 0.5 * np.cos(theta)
    assert np.max(np.abs(heat - heat_expected)) < 1e-12

    # Positive LF Courant number moves a localized state toward increasing
    # spatial index.  Under y[i]=sum_d w[d]x[i+d], its Fourier multiplier has
    # negative imaginary phase: cos(theta) - i*r*sin(theta).
    lf = BASE_ANALYSIS.symbol_from_offsets(
        np.array([0.625, 0.0, 0.375]), offsets, n
    )
    lf_expected = np.cos(theta) - 0.25j * np.sin(theta)
    assert np.max(np.abs(lf - lf_expected)) < 1e-12

    scopes = dict(evaluation_scopes(n // 2))
    assert np.array_equal(scopes["in_support_k1_5"], np.arange(1, 6))
    assert np.array_equal(scopes["near_support_k6_8"], np.arange(6, 9))
    assert np.array_equal(scopes["spectral_extrapolation_k9_nyquist"], np.arange(9, 33))
    assert support_region(1, 32) == "in_support_k1_5"
    assert support_region(5, 32) == "in_support_k1_5"
    assert support_region(6, 32) == "near_support_k6_8"
    assert support_region(9, 32) == "spectral_extrapolation_k9_nyquist"

    exact = selected_symbol_metrics(heat, heat, np.arange(1, 6))
    assert exact["rmse_reference"] < 1e-12
    assert exact["reference_advantage"] > 0.999999

    identity = np.ones_like(heat)
    identity_score = selected_symbol_metrics(identity, heat, np.arange(1, 6))
    assert identity_score["rmse_identity"] < 1e-12
    assert identity_score["reference_advantage"] < -0.999999

    # A one-hot empirical spectrum must reproduce the single-mode error.
    probe = heat.copy()
    probe[3] += 0.125 - 0.25j
    weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    weighted = selected_symbol_metrics(probe, heat, np.arange(1, 6), weights)
    assert abs(weighted["rmse_reference"] - abs(0.125 - 0.25j)) < 1e-12

    # Direct sinusoid eigenfunction check for the corrected sign convention.
    row = np.zeros(n)
    row[offsets % n] = [0.625, 0.0, 0.375]
    matrix = np.stack([np.roll(row, i) for i in range(n)])
    x = np.cos(2 * np.pi * 7 * np.arange(n) / n + 0.37)
    gain = np.fft.fft(matrix @ x)[7] / np.fft.fft(x)[7]
    assert abs(gain - lf[7]) < 1e-12

    print("Support-aware Fourier self-test passed")
    print(f"heat symbol max error: {np.max(np.abs(heat - heat_expected)):.2e}")
    print(f"LF direction max error: {np.max(np.abs(lf - lf_expected)):.2e}")
    print(f"single-mode weighted error: {weighted['rmse_reference']:.6f}")
    print(f"sinusoid gain error: {abs(gain - lf[7]):.2e}")


if __name__ == "__main__":
    main()
