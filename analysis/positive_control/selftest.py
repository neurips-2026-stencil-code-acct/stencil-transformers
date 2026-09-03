"""Deterministic checks for the attention-only positive control."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from common import accumulate_statistics, apply_attention, fit_attention_logits, mean_squared_error


def trajectory_data(stencil: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = np.zeros((12, 13, 32), dtype=np.float32)
    data[:, 0, :] = rng.normal(size=(12, 32))
    for step in range(12):
        data[:, step + 1, :] = apply_attention(data[:, step, :], stencil)
    return data


def check_stencil(stencil: np.ndarray, seed: int) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.npy"
        np.save(path, trajectory_data(stencil, seed))
        statistics = accumulate_statistics(path, chunk_trajectories=4)
        fit = fit_attention_logits(statistics, device="cpu", seed=seed)
        fitted = np.asarray(fit["fitted_weights"])
        maximum_error = float(np.max(np.abs(fitted - stencil)))
        assert maximum_error < 2e-5, (stencil, fitted, maximum_error)
        data = np.load(path)
        inputs = data[:, :-1, :].reshape(-1, data.shape[-1])
        targets = data[:, 1:, :].reshape(-1, data.shape[-1])
        mse = mean_squared_error(inputs, targets, fitted)
        assert mse < 1e-10, mse
        assert np.isclose(fitted.sum(), 1.0, atol=1e-12)
        assert np.all(fitted >= 0)


def main() -> None:
    check_stencil(np.array([0.25, 0.50, 0.25]), 100)
    check_stencil(np.array([0.625, 0.0, 0.375]), 300)
    print("PASS attention-only positive-control self-tests")


if __name__ == "__main__":
    main()

