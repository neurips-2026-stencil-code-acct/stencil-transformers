"""Use the input-to-output DFT sign convention.

This wrapper exists because the original row FFT has the opposite imaginary
sign from Y[k]/X[k] for y[i] = sum_d w[d] x[i+d]. Magnitudes and Jacobian
reference distances are invariant, but LF phase and finite-probe comparisons
need the conjugated row FFT used here.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fourier_analysis as BASE_ANALYSIS


def symbol_from_offsets(values, offsets, n):
    row = BASE_ANALYSIS.embed_offsets(values, offsets, n)
    return np.conj(np.fft.fft(row)[: n // 2 + 1])


BASE_ANALYSIS.symbol_from_offsets = symbol_from_offsets


if __name__ == "__main__":
    BASE_ANALYSIS.main()
