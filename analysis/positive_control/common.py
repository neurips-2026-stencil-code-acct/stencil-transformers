"""Shared implementation for the attention-only positive control.

The model has exactly one spatial mixing operation: a row-stochastic,
three-offset attention kernel.  The analytical stencil is never read by the
fitter; it is loaded only after fitting for held-out evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

# Required by torch deterministic mode for CUDA matrix products. Set this
# before importing torch so direct runs and the launcher share the contract.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


PROTOCOL_VERSION = "1.0.0"
OFFSETS = np.array([-1, 0, 1], dtype=np.int64)
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Condition:
    pde: str
    r: float
    data_dir: str
    seeds: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.pde}_r{self.r:g}"

    @property
    def label(self) -> str:
        prefix = "Heat" if self.pde == "heat" else "LF"
        return f"{prefix} {self.r:.2f}"


CONDITIONS = (
    Condition("heat", 0.10, "heat_new_0.1", tuple(range(100, 120))),
    Condition("heat", 0.25, "heat_new_0.25", tuple(range(100, 120))),
    Condition("heat", 0.40, "heat_new_0.4", tuple(range(100, 120))),
    Condition("lf", 0.10, "lax_friedrichs_r_0.1", tuple(range(300, 320))),
    Condition("lf", 0.25, "lax_friedrichs_r_0.25", tuple(range(300, 320))),
    Condition("lf", 0.40, "lax_friedrichs_r_0.4", tuple(range(300, 320))),
)


@dataclass
class SufficientStatistics:
    gram: np.ndarray
    cross: np.ndarray
    target_sq: float
    n_rows: int
    n_trajectories: int
    n_time_steps: int
    n_space: int
    elapsed_seconds: float

    def normalized(self) -> tuple[np.ndarray, np.ndarray, float]:
        scale = float(self.n_rows)
        return self.gram / scale, self.cross / scale, self.target_sq / scale


def parse_condition_filter(spec: str) -> tuple[Condition, ...]:
    if spec.strip().lower() == "all":
        return CONDITIONS
    requested: set[tuple[str, float]] = set()
    for token in spec.split(","):
        pde, raw_r = token.strip().split(":", 1)
        requested.add((pde.strip().lower(), float(raw_r)))
    selected = tuple(
        condition
        for condition in CONDITIONS
        if any(
            condition.pde == pde and math.isclose(condition.r, r)
            for pde, r in requested
        )
    )
    if len(selected) != len(requested):
        known = ", ".join(f"{item.pde}:{item.r:g}" for item in CONDITIONS)
        raise ValueError(f"Unknown condition in {spec!r}; expected a subset of {known}")
    return selected


def parse_seed_filter(spec: str) -> set[int] | None:
    if spec.strip().lower() == "all":
        return None
    return {int(token.strip()) for token in spec.split(",") if token.strip()}


def task_inventory(
    condition_spec: str = "all",
    seed_spec: str = "all",
    max_models: int | None = None,
) -> list[tuple[Condition, int]]:
    seed_filter = parse_seed_filter(seed_spec)
    tasks: list[tuple[Condition, int]] = []
    for condition in parse_condition_filter(condition_spec):
        for seed in condition.seeds:
            if seed_filter is None or seed in seed_filter:
                tasks.append((condition, seed))
    if max_models is not None:
        tasks = tasks[:max_models]
    if not tasks:
        raise ValueError("The condition/seed filters selected no runs")
    return tasks


def condition_data_dir(data_root: Path, condition: Condition, seed: int) -> Path:
    return data_root / condition.data_dir / f"seed_{seed}"


def accumulate_statistics(
    train_path: Path,
    chunk_trajectories: int = 8,
    max_trajectories: int | None = None,
    max_time_steps: int | None = None,
) -> SufficientStatistics:
    """Accumulate the complete three-offset least-squares objective."""

    array = np.load(train_path, mmap_mode="r")
    if array.ndim != 3:
        raise ValueError(f"{train_path}: expected (trajectory,time,space), got {array.shape}")
    n_trajectories = min(array.shape[0], max_trajectories or array.shape[0])
    available_steps = array.shape[1] - 1
    n_time_steps = min(available_steps, max_time_steps or available_steps)
    n_space = array.shape[2]
    if n_trajectories < 1 or n_time_steps < 1 or n_space < 3:
        raise ValueError(f"{train_path}: insufficient data after limits")

    gram = np.zeros((3, 3), dtype=np.float64)
    cross = np.zeros(3, dtype=np.float64)
    target_sq = 0.0
    n_rows = 0
    started = time.perf_counter()
    for start in range(0, n_trajectories, chunk_trajectories):
        stop = min(start + chunk_trajectories, n_trajectories)
        block = np.asarray(array[start:stop, : n_time_steps + 1, :], dtype=np.float64)
        inputs = block[:, :-1, :]
        targets = block[:, 1:, :]
        features = np.stack(
            (
                np.roll(inputs, 1, axis=-1),
                inputs,
                np.roll(inputs, -1, axis=-1),
            ),
            axis=-1,
        ).reshape(-1, 3)
        target = targets.reshape(-1)
        gram += features.T @ features
        cross += features.T @ target
        target_sq += float(target @ target)
        n_rows += int(features.shape[0])

    return SufficientStatistics(
        gram=gram,
        cross=cross,
        target_sq=target_sq,
        n_rows=n_rows,
        n_trajectories=n_trajectories,
        n_time_steps=n_time_steps,
        n_space=n_space,
        elapsed_seconds=time.perf_counter() - started,
    )


def fit_attention_logits(
    statistics: SufficientStatistics,
    device: str,
    seed: int,
    max_iter: int = 500,
) -> dict[str, object]:
    """Fit three softmax logits without access to the analytical stencil."""

    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.set_device(resolved)
    torch.use_deterministic_algorithms(True)
    generator = np.random.default_rng(seed + 20260828)
    initial_logits = generator.normal(0.0, 0.25, size=3)

    gram_np, cross_np, target_sq = statistics.normalized()
    gram = torch.tensor(gram_np, dtype=torch.float64, device=resolved)
    cross = torch.tensor(cross_np, dtype=torch.float64, device=resolved)
    target_sq_tensor = torch.tensor(target_sq, dtype=torch.float64, device=resolved)
    logits = torch.tensor(
        initial_logits,
        dtype=torch.float64,
        device=resolved,
        requires_grad=True,
    )
    initial_weights = torch.softmax(logits.detach(), dim=0)
    optimizer = torch.optim.LBFGS(
        [logits],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-15,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0

    def objective() -> torch.Tensor:
        weights = torch.softmax(logits, dim=0)
        return weights @ gram @ weights - 2.0 * (weights @ cross) + target_sq_tensor

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        closure_calls += 1
        return loss

    started = time.perf_counter()
    optimizer.step(closure)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        logits -= logits.mean()
        final_weights = torch.softmax(logits, dim=0)
        final_loss = objective()

    if not torch.isfinite(final_loss):
        raise RuntimeError(f"Non-finite fitted objective on {device}")
    return {
        "initial_logits": initial_logits.tolist(),
        "initial_weights": initial_weights.cpu().numpy().tolist(),
        "fitted_logits": logits.detach().cpu().numpy().tolist(),
        "fitted_weights": final_weights.cpu().numpy().tolist(),
        "train_mse_objective": max(float(final_loss.detach().cpu()), 0.0),
        "closure_calls": closure_calls,
        "fit_seconds": elapsed,
    }


def sample_pairs(path: Path, n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.load(path, mmap_mode="r")
    n_trajectories, time_plus_one, n_space = array.shape
    n_time_steps = time_plus_one - 1
    available = n_trajectories * n_time_steps
    count = min(n_pairs, available)
    rng = np.random.default_rng(seed)
    flat = np.sort(rng.choice(available, size=count, replace=False))
    trajectory = flat // n_time_steps
    time_index = flat % n_time_steps
    inputs = np.asarray(array[trajectory, time_index, :], dtype=np.float64)
    targets = np.asarray(array[trajectory, time_index + 1, :], dtype=np.float64)
    if inputs.shape != (count, n_space) or targets.shape != inputs.shape:
        raise AssertionError("Unexpected sampled pair shape")
    return inputs, targets


def apply_attention(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (3,):
        raise ValueError(f"Expected three weights, got {weights.shape}")
    return (
        weights[0] * np.roll(inputs, 1, axis=-1)
        + weights[1] * inputs
        + weights[2] * np.roll(inputs, -1, axis=-1)
    )


def mean_squared_error(inputs: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    residual = apply_attention(inputs, weights) - targets
    return float(np.mean(np.square(residual), dtype=np.float64))


def shape_statistics(weights: np.ndarray) -> dict[str, float]:
    weights = np.asarray(weights, dtype=np.float64)
    flank = weights[0] + weights[2]
    return {
        "centrality": float(weights[1] / weights.sum()),
        "asymmetry": float((weights[2] - weights[0]) / flank) if flank > 0 else math.nan,
    }


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def bootstrap_mean(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or len(values) == 0:
        return math.nan, math.nan, math.nan
    draws = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )
