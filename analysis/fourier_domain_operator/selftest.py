"""Small exact tests for Fourier symbol extraction and sinusoid conventions."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fourier_analysis import symbol_from_offsets, symbol_metrics


def main():
    n = 64
    d = np.array([-1, 0, 1])
    k = np.arange(n // 2 + 1)
    theta = 2 * np.pi * k / n

    heat = symbol_from_offsets(np.array([0.25, 0.5, 0.25]), d, n)
    heat_expected = 0.5 + 0.5 * np.cos(theta)
    assert np.max(np.abs(heat - heat_expected)) < 1e-12

    lf = symbol_from_offsets(np.array([0.625, 0.0, 0.375]), d, n)
    lf_expected = np.cos(theta) + 0.25j * np.sin(theta)
    assert np.max(np.abs(lf - lf_expected)) < 1e-12

    exact = symbol_metrics(heat, heat)
    identity = symbol_metrics(np.ones_like(heat), heat)
    assert exact["rmse_reference"] < 1e-12
    assert exact["reference_advantage"] > 0.999999
    assert identity["rmse_identity"] < 1e-12
    assert identity["reference_advantage"] < -0.999999

    # Direct sinusoid eigenfunction check for the same embedded heat row.
    row = np.zeros(n)
    row[d % n] = [0.25, 0.5, 0.25]
    matrix = np.stack([np.roll(row, i) for i in range(n)])
    x = np.cos(2 * np.pi * 7 * np.arange(n) / n + 0.37)
    z = np.fft.fft(matrix @ x)[7] / np.fft.fft(x)[7]
    assert abs(z - heat[7]) < 1e-12

    print("Fourier-analysis self-test passed")
    print(f"heat symbol max error: {np.max(np.abs(heat - heat_expected)):.2e}")
    print(f"LF symbol max error:   {np.max(np.abs(lf - lf_expected)):.2e}")
    print(f"sinusoid gain error:   {abs(z - heat[7]):.2e}")


if __name__ == "__main__":
    main()
