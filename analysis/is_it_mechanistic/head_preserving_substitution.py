"""Freeze attention without averaging different heads together.

For each checkpoint, this script:

1. computes one mean attention matrix per layer and head from validation inputs;
2. evaluates those fixed, head-specific matrices on test inputs;
3. compares them with a validation-derived layer mean shared by all heads and
   with the known finite-difference stencil; and
4. checks that the patched forward pass matches the ordinary model when no
   attention matrix is replaced.

The head-specific control removes changes across inputs while retaining the
different mean matrix learned by each head.  The layer-mean control removes
both changes across inputs and differences among heads.  The stencil control
asks whether the numerical update can serve as a fixed replacement inside the
already-trained network; it does not test whether the network learned that
stencil during training.

All replacements occur after the attention softmax and before multiplication
by the value vectors.  The trained value projections, output projections,
residual paths, layer normalisation, and feed-forward blocks remain unchanged.

This is an isolated review analysis. It does not modify mechanistic_model.py,
model_review_fix.py, attention_substitution.py, checkpoints, or data.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ATTENTION_ANALYSIS_DIR = HERE.parent / "what_do_heads_learn"
sys.path.insert(0, str(ATTENTION_ANALYSIS_DIR))
sys.path.insert(0, str(HERE))

import attention_common as C
from build_predictors import CONDITION_REGEX
from extract_attention_profiles import condition_data_dir
from model_review_fix import (
    AttentionControl,
    baseline_mse,
    circulant_from_profile,
    load_model,
    model_output,
    skill_guard,
    to_batch,
)


SPLIT_FILENAMES = {
    "validation": ("val.npy", "validation.npy", "valid.npy"),
    "test": ("test.npy", "test_u.npy", "u_test.npy"),
}


class CheckedAttentionControl(AttentionControl):
    """AttentionControl with an explicit count of completed replacements."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_substituted = 0

    def _sub_for(self, idx, n, device, dtype):
        substitute = super()._sub_for(idx, n, device, dtype)
        if substitute is not None:
            self.n_substituted += 1
        return substitute

    def assert_substituted(self):
        if self.substitute is not None and self.n_substituted == 0:
            raise RuntimeError(
                "attention was intercepted, but the requested matrix was not "
                "used in any selected layer"
            )


def resolve_split_path(data_dir: str | os.PathLike, split: str) -> Path:
    """Return an explicit validation or test file; never fall back to training."""
    try:
        names = SPLIT_FILENAMES[split]
    except KeyError as exc:
        raise ValueError(f"unknown split {split!r}") from exc
    root = Path(data_dir)
    for name in names:
        path = root / name
        if path.is_file():
            return path.resolve()
    expected = ", ".join(str(root / name) for name in names)
    raise FileNotFoundError(f"no {split} array found; tried {expected}")


def assert_distinct_split_paths(validation_path: Path, test_path: Path) -> None:
    """Refuse to estimate and evaluate frozen attention on the same file."""
    validation_path = Path(validation_path).resolve()
    test_path = Path(test_path).resolve()
    same = validation_path == test_path
    if validation_path.exists() and test_path.exists():
        same = same or os.path.samefile(validation_path, test_path)
    if same:
        raise ValueError(
            "validation and test resolve to the same file; frozen attention "
            "must be estimated on validation inputs and evaluated on test inputs"
        )


def load_split_pairs(
    path: str | os.PathLike,
    n_pairs: int,
    n_input_steps: int = 1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample consecutive input-target pairs from one named trajectory split.

    The array must have shape (trajectory, time, space).  Sampling is performed
    directly from the memory-mapped array, avoiding construction of every
    possible pair before retaining the requested subset.
    """
    path = Path(path)
    arr = np.asarray(np.load(path, mmap_mode="r"))
    if arr.ndim != 3:
        raise ValueError(
            f"{path} has shape {arr.shape}; expected (trajectory, time, space)"
        )
    if not 1 <= n_input_steps < arr.shape[1]:
        raise ValueError(
            f"n_input_steps must be between 1 and {arr.shape[1] - 1}; "
            f"received {n_input_steps}"
        )

    n_trajectories, n_times, _ = arr.shape
    n_available = n_trajectories * (n_times - n_input_steps)
    n_take = min(int(n_pairs), n_available)
    if n_take < 1:
        raise ValueError("n_pairs must be positive")

    rng = np.random.default_rng(seed)
    flat = np.sort(rng.choice(n_available, size=n_take, replace=False))
    trajectory = flat % n_trajectories
    target_time = flat // n_trajectories + n_input_steps
    history = target_time[:, None] - n_input_steps + np.arange(n_input_steps)[None]

    x = np.asarray(arr[trajectory[:, None], history, :], dtype=np.float32)
    y = np.asarray(arr[trajectory, target_time, :], dtype=np.float32)
    return x, y


def validate_mean_attention(mean: np.ndarray, n_space: int) -> None:
    """Check the expected (layer, head, query, source) shape and row sums."""
    if mean.ndim != 4:
        raise ValueError(
            "head-preserving mean attention must have shape "
            f"(layer, head, query, source); received {mean.shape}"
        )
    if mean.shape[-2:] != (n_space, n_space):
        raise ValueError(
            f"attention matrices have shape {mean.shape[-2:]}; expected "
            f"({n_space}, {n_space})"
        )
    if not np.isfinite(mean).all():
        raise ValueError("mean attention contains non-finite values")
    if (mean < -1e-7).any():
        raise ValueError("mean attention contains negative entries")
    if not np.allclose(mean.sum(axis=-1), 1.0, atol=2e-5, rtol=2e-5):
        raise ValueError("mean attention rows do not sum to one")


def head_preserving_mean_attention(model, x, device, batch_size=256):
    """Average over validation inputs while keeping layers and heads separate."""
    import torch

    sums: list[np.ndarray] | None = None
    count = 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = to_batch(x[start:start + batch_size], device)
            control = CheckedAttentionControl(capture=True)
            with control:
                model_output(model, xb)
            control.assert_intercepted()
            captured = [np.asarray(a, dtype=np.float64) for a in control.attentions]
            if sums is None:
                sums = [np.zeros_like(a, dtype=np.float64) for a in captured]
            if len(captured) != len(sums):
                raise RuntimeError(
                    "the number of intercepted attention layers changed between batches"
                )
            for layer, attention in enumerate(captured):
                if attention.shape != sums[layer].shape:
                    raise RuntimeError(
                        "the captured attention shape changed between validation batches"
                    )
                sums[layer] += attention * len(xb)
            count += len(xb)

    if sums is None or count == 0:
        raise RuntimeError("no validation attention was collected")
    mean = np.stack([total / count for total in sums], axis=0)
    validate_mean_attention(mean, x.shape[-1])
    return mean


def evaluate(model, x, y, device, batch_size=256, substitute=None, layers=None):
    """Evaluate test MSE and verify interception and replacement in every batch."""
    import torch

    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = to_batch(x[start:start + batch_size], device)
            yb = torch.from_numpy(y[start:start + batch_size]).to(device)
            if substitute is None:
                prediction = model_output(model, xb)
            else:
                control = CheckedAttentionControl(substitute=substitute, layers=layers)
                with control:
                    prediction = model_output(model, xb)
                control.assert_intercepted()
                control.assert_substituted()
            total += float(((prediction - yb) ** 2).sum())
            count += yb.numel()
    return total / max(count, 1)


def verify_patched_noop(model, x, device, batch_size=8):
    """Confirm that interception alone leaves the model output unchanged."""
    import torch

    xb = to_batch(x[: min(batch_size, len(x))], device)
    with torch.no_grad():
        ordinary = model_output(model, xb)
        control = CheckedAttentionControl()
        with control:
            patched = model_output(model, xb)
        control.assert_intercepted()

    difference = (ordinary - patched).abs()
    max_abs = float(difference.max().item())
    scale = float(ordinary.abs().max().item())
    tolerance = 1e-6 + 1e-5 * scale
    if max_abs > tolerance:
        raise RuntimeError(
            "the patched no-replacement forward does not reproduce the ordinary "
            f"model output: max absolute difference {max_abs:.3e}, allowed "
            f"{tolerance:.3e}"
        )
    return max_abs


def analytical_stencil(predictors, key: str, n_space: int) -> np.ndarray:
    """Build the row-stochastic circulant matrix used by the original run."""
    weights_key = f"{key}/stencil_raw"
    offsets_key = f"{key}/stencil_offsets"
    if weights_key not in predictors or offsets_key not in predictors:
        raise KeyError(f"analytical stencil is missing for {key}")
    weights = np.asarray(predictors[weights_key], dtype=np.float64)
    offsets = np.asarray(predictors[offsets_key], dtype=int)
    matrix = circulant_from_profile(np.abs(weights), offsets, n_space)
    if not np.allclose(matrix.sum(axis=-1), 1.0, atol=1e-12):
        raise ValueError(f"stencil substitute for {key} is not row-stochastic")
    return matrix


def write_rows(rows: list[dict], output_csv: Path) -> None:
    """Write restartable results without leaving a partially written CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, output_csv)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--runs", required=True, help="recursive checkpoint glob")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--data-template", default="{pde}_r{r}")
    parser.add_argument("--condition-regex", default=CONDITION_REGEX)
    parser.add_argument(
        "--predictors", default="analysis/what_do_heads_learn/predictors.npz"
    )
    parser.add_argument(
        "--out", default="analysis/is_it_mechanistic/results/head_preserving"
    )
    parser.add_argument("--pdes", default="heat,lf")
    parser.add_argument("--n-validation-pairs", type=int, default=2048)
    parser.add_argument("--n-test-pairs", type=int, default=2048)
    parser.add_argument("--n-input-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model-factory", default="model_adapter:build_model_from_checkpoint"
    )
    parser.add_argument("--run-regex", default=C.DEFAULT_RUN_REGEX)
    parser.add_argument("--sample-seed", type=int, default=20260824)
    parser.add_argument("--expected-models", type=int, default=120)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--per-layer", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    allowed_pdes = {item.strip() for item in args.pdes.split(",") if item.strip()}
    parsed = []
    for path in sorted(glob.glob(args.runs, recursive=True)):
        meta = C.parse_run(path, args.run_regex)
        if meta is not None and meta.pde in allowed_pdes:
            parsed.append((path, meta))
    if not parsed:
        raise SystemExit(f"no heat/Lax--Friedrichs checkpoints matched {args.runs}")
    if args.max_models is None and args.expected_models > 0:
        if len(parsed) != args.expected_models:
            raise SystemExit(
                f"found {len(parsed)} workshop checkpoints; expected "
                f"{args.expected_models}. Refusing a silent partial run."
            )
    if args.max_models is not None:
        parsed = parsed[: args.max_models]

    predictors = np.load(args.predictors)
    output_dir = Path(args.out)
    output_csv = output_dir / "substitution.csv"
    rows: list[dict] = []
    completed: set[str] = set()
    if output_csv.exists():
        if not args.resume:
            raise SystemExit(
                f"{output_csv} already exists; use --resume or choose a new --out"
            )
        previous = pd.read_csv(output_csv)
        rows = previous.to_dict("records")
        if "checkpoint" in previous:
            completed = set(previous["checkpoint"].astype(str))

    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "resolved_models": len(parsed),
            "allowed_pdes": sorted(allowed_pdes),
            "calibration_split": "validation",
            "evaluation_split": "test",
        }
    )
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    for number, (checkpoint, meta) in enumerate(parsed, start=1):
        checkpoint_id = str(Path(checkpoint).resolve())
        if checkpoint_id in completed:
            print(f"[{number}/{len(parsed)}] already complete: {checkpoint}", flush=True)
            continue

        key = f"{meta.pde}_r{meta.r:g}"
        data_dir = condition_data_dir(
            args.data_root,
            meta,
            args.data_template,
            args.condition_regex,
        )
        validation_path = resolve_split_path(data_dir, "validation")
        test_path = resolve_split_path(data_dir, "test")
        assert_distinct_split_paths(validation_path, test_path)

        validation_x, _ = load_split_pairs(
            validation_path,
            args.n_validation_pairs,
            args.n_input_steps,
            seed=args.sample_seed + meta.seed,
        )
        test_x, test_y = load_split_pairs(
            test_path,
            args.n_test_pairs,
            args.n_input_steps,
            seed=args.sample_seed + 1_000_003 + meta.seed,
        )

        model = load_model(checkpoint, args.device, args.model_factory)
        no_op_max_abs = verify_patched_noop(model, test_x, args.device)
        mean_by_head = head_preserving_mean_attention(
            model, validation_x, args.device, args.batch_size
        )
        mean_by_layer = mean_by_head.mean(axis=1)
        stencil = analytical_stencil(predictors, key, test_x.shape[-1])

        n_layers, n_heads = mean_by_head.shape[:2]
        head_dispersion = float(
            np.mean(np.abs(mean_by_head - mean_by_layer[:, None]))
        )
        original_mse = evaluate(
            model, test_x, test_y, args.device, args.batch_size
        )
        persistence_mse = baseline_mse(test_x, test_y)
        skill = skill_guard(
            original_mse,
            persistence_mse,
            f"{meta.pde} r={meta.r:g} seed {meta.seed}: ",
        )

        substitutes = {
            "frozen_per_head_validation": (mean_by_head, True),
            "frozen_layer_mean_validation": (mean_by_layer, False),
            "analytical_stencil": (stencil, False),
        }
        targets = [(None, "all")]
        if args.per_layer:
            targets.extend(({layer}, f"layer{layer}") for layer in range(n_layers))

        model_rows = []
        common = {
            "pde": meta.pde,
            "r": meta.r,
            "seed": meta.seed,
            "checkpoint": checkpoint_id,
            "validation_path": str(validation_path),
            "test_path": str(test_path),
            "n_validation_pairs": len(validation_x),
            "n_test_pairs": len(test_x),
            "n_layers": n_layers,
            "n_heads": n_heads,
            "head_dispersion": head_dispersion,
            "patched_noop_max_abs": no_op_max_abs,
            "mse_original": original_mse,
            "mse_persistence": persistence_mse,
            "skill": skill,
        }
        denominator = max(persistence_mse - original_mse, 1e-12)
        for layers, target in targets:
            for name, (matrix, preserves_heads) in substitutes.items():
                mse = evaluate(
                    model,
                    test_x,
                    test_y,
                    args.device,
                    args.batch_size,
                    substitute=matrix,
                    layers=layers,
                )
                model_rows.append(
                    {
                        **common,
                        "target": target,
                        "substitution": name,
                        "preserves_head_identity": preserves_heads,
                        "calibration_split": "validation",
                        "evaluation_split": "test",
                        "mse": mse,
                        "excess": (mse - original_mse) / denominator,
                        "ratio": mse / max(original_mse, 1e-30),
                    }
                )
            model_rows.append(
                {
                    **common,
                    "target": target,
                    "substitution": "original",
                    "preserves_head_identity": True,
                    "calibration_split": "none",
                    "evaluation_split": "test",
                    "mse": original_mse,
                    "excess": 0.0,
                    "ratio": 1.0,
                }
            )

        rows.extend(model_rows)
        write_rows(rows, output_csv)
        print(
            f"[{number}/{len(parsed)}] {meta.pde} r={meta.r:g} seed={meta.seed}: "
            f"head-frozen ratio={model_rows[0]['ratio']:.4g}, "
            f"layer-mean ratio={model_rows[1]['ratio']:.4g}, "
            f"stencil ratio={model_rows[2]['ratio']:.4g}",
            flush=True,
        )
        del model

    frame = pd.DataFrame(rows)
    summary = (
        frame[frame["target"] == "all"]
        .groupby(["pde", "r", "substitution"], as_index=False)
        .agg(
            n_models=("seed", "nunique"),
            mean_mse=("mse", "mean"),
            mean_ratio=("ratio", "mean"),
            median_ratio=("ratio", "median"),
            mean_excess=("excess", "mean"),
        )
    )
    summary.to_csv(output_dir / "condition_summary.csv", index=False)
    print(f"wrote {output_csv}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
