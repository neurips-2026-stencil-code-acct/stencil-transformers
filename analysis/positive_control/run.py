"""Fit and aggregate the 120 attention-only positive controls."""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import torch

from common import (
    CONDITIONS,
    OFFSETS,
    PROTOCOL_VERSION,
    ROOT,
    accumulate_statistics,
    atomic_write_json,
    bootstrap_mean,
    condition_data_dir,
    fit_attention_logits,
    mean_squared_error,
    parse_condition_filter,
    shape_statistics,
    task_inventory,
    sample_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="analysis/positive_control/results")
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--expected-models", type=int, default=120)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-time-steps", type=int, default=None)
    parser.add_argument("--chunk-trajectories", type=int, default=8)
    parser.add_argument("--evaluation-pairs", type=int, default=2048)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def model_path(output: Path, condition, seed: int) -> Path:
    return output / "models" / f"{condition.pde}_r{condition.r:g}_seed{seed}.json"


def run_one(
    condition,
    seed: int,
    device: str,
    data_root: Path,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    started = time.perf_counter()
    source = condition_data_dir(data_root, condition, seed)
    train_path = source / "train.npy"
    validation_path = source / "val.npy"
    test_path = source / "test.npy"
    stencil_path = source / "stencil.npy"
    for path in (train_path, validation_path, test_path, stencil_path):
        if not path.exists():
            raise FileNotFoundError(path)

    statistics = accumulate_statistics(
        train_path,
        chunk_trajectories=args.chunk_trajectories,
        max_trajectories=args.max_trajectories,
        max_time_steps=args.max_time_steps,
    )
    fit = fit_attention_logits(statistics, device=device, seed=seed)
    fitted = np.asarray(fit["fitted_weights"], dtype=np.float64)
    initial = np.asarray(fit["initial_weights"], dtype=np.float64)

    # The analytical stencil is deliberately loaded only after fitting.
    stencil = np.load(stencil_path).ravel().astype(np.float64)
    if stencil.shape != (3,):
        raise ValueError(f"{stencil_path}: expected three coefficients, got {stencil.shape}")
    validation_x, validation_y = sample_pairs(
        validation_path,
        args.evaluation_pairs,
        seed=seed + 101_000,
    )
    test_x, test_y = sample_pairs(
        test_path,
        args.evaluation_pairs,
        seed=seed + 202_000,
    )
    persistence = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    mse_validation = mean_squared_error(validation_x, validation_y, fitted)
    mse_fitted = mean_squared_error(test_x, test_y, fitted)
    mse_initial = mean_squared_error(test_x, test_y, initial)
    mse_stencil = mean_squared_error(test_x, test_y, stencil)
    mse_persistence = mean_squared_error(test_x, test_y, persistence)
    denominator = mse_persistence - mse_fitted
    if denominator <= 0:
        raise RuntimeError(
            f"{condition.key} seed {seed}: fitted model does not beat persistence"
        )
    coefficient_error = fitted - stencil
    fitted_shape = shape_statistics(fitted)
    stencil_shape = shape_statistics(stencil)

    resolved_device = torch.device(device)
    device_name = (
        torch.cuda.get_device_name(resolved_device)
        if resolved_device.type == "cuda"
        else "CPU"
    )
    payload: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "pde": condition.pde,
        "r": condition.r,
        "seed": seed,
        "condition": condition.key,
        "device": device,
        "device_name": device_name,
        "data_dir": str(source.resolve()),
        "train_path": str(train_path.resolve()),
        "validation_path": str(validation_path.resolve()),
        "test_path": str(test_path.resolve()),
        "stencil_loaded_after_fit": True,
        "offsets": OFFSETS.tolist(),
        "train_rows": statistics.n_rows,
        "train_trajectories": statistics.n_trajectories,
        "train_time_steps": statistics.n_time_steps,
        "n_space": statistics.n_space,
        "statistics_seconds": statistics.elapsed_seconds,
        "evaluation_pairs": len(test_x),
        **fit,
        "stencil_weights": stencil.tolist(),
        "coefficient_error": coefficient_error.tolist(),
        "coefficient_l1": float(np.abs(coefficient_error).sum()),
        "coefficient_l2": float(np.linalg.norm(coefficient_error)),
        "coefficient_max_abs": float(np.abs(coefficient_error).max()),
        "initial_coefficient_l2": float(np.linalg.norm(initial - stencil)),
        "centrality": fitted_shape["centrality"],
        "asymmetry": fitted_shape["asymmetry"],
        "target_centrality": stencil_shape["centrality"],
        "target_asymmetry": stencil_shape["asymmetry"],
        "mse_validation": mse_validation,
        "mse_fitted": mse_fitted,
        "mse_fixed": mse_fitted,
        "mse_initial": mse_initial,
        "mse_stencil": mse_stencil,
        "mse_persistence": mse_persistence,
        "skill": 1.0 - mse_fitted / mse_persistence,
        "excess_initial": (mse_initial - mse_fitted) / denominator,
        "excess_stencil": (mse_stencil - mse_fitted) / denominator,
        "total_seconds": time.perf_counter() - started,
    }
    atomic_write_json(model_path(output, condition, seed), payload)
    return payload


def load_model_payloads(output: Path) -> list[dict[str, object]]:
    paths = sorted((output / "models").glob("*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def seed_frame(payloads: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        fitted = payload["fitted_weights"]
        initial = payload["initial_weights"]
        stencil = payload["stencil_weights"]
        rows.append(
            {
                "pde": payload["pde"],
                "r": payload["r"],
                "seed": payload["seed"],
                "condition": payload["condition"],
                "device": payload["device"],
                "device_name": payload["device_name"],
                "weight_minus": fitted[0],
                "weight_center": fitted[1],
                "weight_plus": fitted[2],
                "initial_minus": initial[0],
                "initial_center": initial[1],
                "initial_plus": initial[2],
                "stencil_minus": stencil[0],
                "stencil_center": stencil[1],
                "stencil_plus": stencil[2],
                "coefficient_l1": payload["coefficient_l1"],
                "coefficient_l2": payload["coefficient_l2"],
                "coefficient_max_abs": payload["coefficient_max_abs"],
                "initial_coefficient_l2": payload["initial_coefficient_l2"],
                "centrality": payload["centrality"],
                "asymmetry": payload["asymmetry"],
                "target_centrality": payload["target_centrality"],
                "target_asymmetry": payload["target_asymmetry"],
                "mse_fitted": payload["mse_fitted"],
                "mse_fixed": payload["mse_fixed"],
                "mse_initial": payload["mse_initial"],
                "mse_stencil": payload["mse_stencil"],
                "mse_persistence": payload["mse_persistence"],
                "skill": payload["skill"],
                "excess_initial": payload["excess_initial"],
                "excess_stencil": payload["excess_stencil"],
                "train_rows": payload["train_rows"],
                "statistics_seconds": payload["statistics_seconds"],
                "fit_seconds": payload["fit_seconds"],
                "total_seconds": payload["total_seconds"],
            }
        )
    return pd.DataFrame(rows).sort_values(["pde", "r", "seed"]).reset_index(drop=True)


def assign_attention_labels(frame: pd.DataFrame) -> pd.DataFrame:
    attention_dir = ROOT / "analysis" / "what_do_heads_learn"
    sys.path.insert(0, str(attention_dir))
    import assign_heads  # type: ignore
    import attention_common  # type: ignore

    predictors = np.load(attention_dir / "predictors.npz")
    runs = []
    n_space = 64
    offsets = attention_common.offsets_full(n_space)
    for row in frame.itertuples(index=False):
        ring = np.zeros(n_space, dtype=np.float64)
        ring[OFFSETS % n_space] = np.array(
            [row.weight_minus, row.weight_center, row.weight_plus]
        )
        ordered = ring[offsets % n_space]
        runs.append(
            {
                "pde": row.pde,
                "r": row.r,
                "seed": row.seed,
                "path": f"positive_control/{row.condition}/seed_{row.seed}",
                "offsets": offsets,
                "profile": ordered[None, None, :],
            }
        )
    trained = assign_heads.head_records(
        runs,
        predictors,
        half_width=4,
        divergence="js",
        is_baseline=False,
        min_structure=0.01,
    )
    baseline = pd.read_csv(attention_dir / "results" / "baseline_heads.csv")
    conditions = sorted({f"{row.pde}_r{row.r:g}" for row in frame.itertuples()})
    offsets_by_condition = {
        condition: offsets for condition in conditions
    }
    separation = assign_heads.predictor_separation(
        predictors,
        conditions,
        offsets_by_condition,
        half_width=4,
        divergence="js",
        min_structure=0.01,
    )
    unidentifiable = tuple(
        (condition.rsplit("_r", 1)[0], float(condition.rsplit("_r", 1)[1]))
        for condition in separation.loc[
            separation["d_acf_spectral"] < 0.01, "condition"
        ]
    )
    assigned = assign_heads.assign(
        trained,
        baseline,
        gate_pct=5.0,
        floor_pct=5.0,
        spectral_margin=0.05,
        merge_unidentifiable=unidentifiable,
    )
    return assigned.sort_values(["pde", "r", "seed"]).reset_index(drop=True)


def condition_bootstrap(frame: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    metrics = [
        "weight_minus",
        "weight_center",
        "weight_plus",
        "coefficient_l2",
        "coefficient_max_abs",
        "centrality",
        "asymmetry",
        "mse_fitted",
        "mse_stencil",
        "mse_persistence",
        "skill",
        "excess_stencil",
        "total_seconds",
    ]
    rng = np.random.default_rng(20260828)
    rows = []
    for (pde, r), group in frame.groupby(["pde", "r"], sort=True):
        for metric in metrics:
            mean, lo, hi = bootstrap_mean(
                group[metric].to_numpy(),
                rng,
                n_bootstrap,
            )
            rows.append(
                {
                    "pde": pde,
                    "r": r,
                    "metric": metric,
                    "mean": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_seeds": len(group),
                }
            )
    return pd.DataFrame(rows)


def tracking_regression(frame: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260829)
    rows = []
    settings = (
        ("heat", "centrality", "target_centrality"),
        ("lf", "asymmetry", "target_asymmetry"),
    )
    for pde, measured_column, target_column in settings:
        part = frame[frame["pde"] == pde].copy()
        if part.empty:
            continue
        response = part.pivot(
            index="seed",
            columns="r",
            values=measured_column,
        ).sort_index(axis=1)
        targets = (
            part.groupby("r")[target_column]
            .first()
            .reindex(response.columns)
            .to_numpy(dtype=np.float64)
        )
        if response.isna().any().any():
            raise RuntimeError(f"{pde}: incomplete seed-by-condition response")
        design = np.column_stack((np.ones(len(targets)), targets))
        projection = np.linalg.pinv(design)
        seed_values = response.to_numpy(dtype=np.float64)
        observed = projection @ seed_values.mean(axis=0)
        sampled = rng.integers(
            0,
            len(seed_values),
            size=(n_bootstrap, len(seed_values)),
        )
        condition_means = seed_values[sampled].mean(axis=1)
        draws = condition_means @ projection.T
        seeds = response.index.to_numpy()
        for term_index, term in enumerate(("intercept", "slope")):
            rows.append(
                {
                    "pde": pde,
                    "statistic": measured_column,
                    "term": term,
                    "coef": observed[term_index],
                    "ci_lo": np.quantile(draws[:, term_index], 0.025),
                    "ci_hi": np.quantile(draws[:, term_index], 0.975),
                    "n_seeds": len(seeds),
                    "n_conditions": part["r"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def aggregate(output: Path, n_bootstrap: int) -> None:
    payloads = load_model_payloads(output)
    frame = seed_frame(payloads)
    frame.to_csv(output / "seed_metrics.csv", index=False)
    assigned = assign_attention_labels(frame)
    assigned.to_csv(output / "attention_assignment.csv", index=False)
    labels = assigned[["pde", "r", "seed", "label"]]
    frame = frame.merge(labels, on=["pde", "r", "seed"], how="left", validate="one_to_one")
    frame.to_csv(output / "seed_metrics.csv", index=False)
    condition_bootstrap(frame, n_bootstrap).to_csv(
        output / "condition_bootstrap_summary.csv",
        index=False,
    )
    tracking_regression(frame, n_bootstrap).to_csv(
        output / "tracking_regression.csv",
        index=False,
    )


def progress_bar(done: int, total: int, started: float, device_counts: dict[str, int], last: str) -> str:
    width = 32
    filled = int(width * done / total)
    bar = "=" * filled + ">" + "." * max(width - filled - 1, 0) if done < total else "=" * width
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float("inf")
    eta_text = "--:--" if not np.isfinite(eta) else time.strftime("%M:%S", time.gmtime(eta))
    counts = " ".join(f"{device}={count}" for device, count in device_counts.items())
    return f"[{bar}] {done:3d}/{total} elapsed={elapsed:6.1f}s eta={eta_text} {counts} last={last}"


def main() -> None:
    args = parse_args()
    data_root = (ROOT / args.data_root).resolve()
    output = (ROOT / args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = task_inventory(args.conditions, args.seeds, args.max_models)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("At least one device is required")
    for device in devices:
        resolved = torch.device(device)
        if resolved.type == "cuda":
            if not torch.cuda.is_available() or resolved.index is None or resolved.index >= torch.cuda.device_count():
                raise RuntimeError(f"Unavailable CUDA device: {device}")
            probe = torch.ones(16, device=resolved)
            if float(probe.sum().cpu()) != 16.0:
                raise RuntimeError(f"CUDA probe failed: {device}")

    pending = []
    completed_payloads: list[dict[str, object]] = []
    for index, (condition, seed) in enumerate(tasks):
        path = model_path(output, condition, seed)
        if args.resume and path.exists():
            completed_payloads.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append((condition, seed, devices[index % len(devices)]))

    started = time.perf_counter()
    device_counts = {device: 0 for device in devices}
    for payload in completed_payloads:
        device_counts[str(payload["device"])] = device_counts.get(str(payload["device"]), 0) + 1
    done = len(completed_payloads)
    print(
        f"Positive-control run: total={len(tasks)} resume={done} pending={len(pending)} "
        f"devices={','.join(devices)}",
        flush=True,
    )
    if done:
        print(progress_bar(done, len(tasks), started, device_counts, "resumed"), flush=True)

    executors = {device: ThreadPoolExecutor(max_workers=1, thread_name_prefix=device.replace(":", "")) for device in devices}
    futures = {}
    try:
        for condition, seed, device in pending:
            future = executors[device].submit(
                run_one,
                condition,
                seed,
                device,
                data_root,
                output,
                args,
            )
            futures[future] = (condition, seed, device)
        for future in as_completed(futures):
            condition, seed, device = futures[future]
            payload = future.result()
            done += 1
            device_counts[device] += 1
            last = f"{condition.key}/seed{seed} err={payload['coefficient_max_abs']:.2e}"
            print(progress_bar(done, len(tasks), started, device_counts, last), flush=True)
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=False)

    aggregate(output, args.bootstrap)
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        git_status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=ROOT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        git_head, git_status = "unavailable", "unavailable"
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "command": sys.argv,
        "arguments": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "devices": {
            device: torch.cuda.get_device_name(torch.device(device))
            for device in devices
            if torch.device(device).type == "cuda"
        },
        "n_models": len(tasks),
        "device_counts": device_counts,
        "elapsed_seconds": time.perf_counter() - started,
        "git_head": git_head,
        "git_status_short": git_status,
        "rollback_command": (
            r"& .\.venv\Scripts\python.exe -B -m pip install --force-reinstall "
            r"--no-deps torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118"
        ),
    }
    atomic_write_json(output / "run_config.json", config)
    if args.expected_models and len(tasks) != args.expected_models:
        print(
            f"NOTE expected-models={args.expected_models}, selected={len(tasks)}; "
            "this is appropriate only for a declared preflight.",
            flush=True,
        )
    print(f"Completed {len(tasks)} fits in {config['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
