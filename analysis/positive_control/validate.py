"""Fail-closed structural validation for positive-control results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import CONDITIONS, PROTOCOL_VERSION, ROOT, atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--results", default="analysis/positive_control/results")
    parser.add_argument("--expected-models", type=int, default=120)
    parser.add_argument("--require-both-gpus", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = (ROOT / args.results).resolve()
    frame = pd.read_csv(results / "seed_metrics.csv")
    assignment = pd.read_csv(results / "attention_assignment.csv")
    condition_summary = pd.read_csv(results / "condition_bootstrap_summary.csv")
    regression = pd.read_csv(results / "tracking_regression.csv")
    config = json.loads((results / "run_config.json").read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("model_count", len(frame) == args.expected_models, f"{len(frame)}/{args.expected_models}")
    check(
        "unique_models",
        not frame.duplicated(["pde", "r", "seed"]).any(),
        "one row per condition and seed",
    )
    check(
        "assignment_count",
        len(assignment) == len(frame),
        f"{len(assignment)}/{len(frame)}",
    )
    check(
        "protocol_version",
        config.get("protocol_version") == PROTOCOL_VERSION,
        str(config.get("protocol_version")),
    )
    expected_conditions = {(item.pde, item.r) for item in CONDITIONS}
    actual_conditions = {(row.pde, float(row.r)) for row in frame.itertuples()}
    if args.expected_models == 120:
        check("six_conditions", actual_conditions == expected_conditions, str(sorted(actual_conditions)))
        counts = frame.groupby(["pde", "r"])["seed"].nunique()
        check("twenty_seeds_per_condition", bool((counts == 20).all()), counts.to_dict().__repr__())
    finite_columns = [
        "weight_minus",
        "weight_center",
        "weight_plus",
        "coefficient_l2",
        "mse_fitted",
        "mse_stencil",
        "mse_persistence",
        "skill",
    ]
    check(
        "finite_primary_metrics",
        bool(np.isfinite(frame[finite_columns].to_numpy()).all()),
        ",".join(finite_columns),
    )
    weight_sum = frame[["weight_minus", "weight_center", "weight_plus"]].sum(axis=1)
    check("row_stochastic", bool(np.allclose(weight_sum, 1.0, atol=1e-10)), f"max error={np.max(np.abs(weight_sum - 1)):.3e}")
    check(
        "nonnegative_attention",
        bool((frame[["weight_minus", "weight_center", "weight_plus"]] >= 0).all().all()),
        "all fitted weights nonnegative",
    )
    check(
        "fixed_is_noop",
        bool(np.allclose(frame["mse_fixed"], frame["mse_fitted"], rtol=0, atol=0)),
        "stored fixed attention equals fitted attention",
    )
    stencil_tolerance = 1e-10 * frame["mse_persistence"] + 1e-14
    check(
        "analytical_generator_consistency",
        bool((frame["mse_stencil"] <= stencil_tolerance).all()),
        f"max stencil MSE={frame['mse_stencil'].max():.3e}",
    )
    check(
        "fitted_beats_persistence",
        bool((frame["mse_fitted"] < frame["mse_persistence"]).all()),
        f"minimum skill={frame['skill'].min():.6f}",
    )
    check(
        "fitting_improves_coefficients",
        bool((frame["coefficient_l2"] < frame["initial_coefficient_l2"]).all()),
        "all seeds improve over initialization",
    )
    model_files = list((results / "models").glob("*.json"))
    check("model_artifact_count", len(model_files) == len(frame), f"{len(model_files)}/{len(frame)}")
    loaded_after = []
    for path in model_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded_after.append(payload.get("stencil_loaded_after_fit") is True)
    check("no_stencil_fit_input", all(loaded_after), "stencil loaded only after fitting")
    if args.require_both_gpus:
        devices = set(frame["device"])
        check("both_gpus_used", {"cuda:0", "cuda:1"}.issubset(devices), str(sorted(devices)))
    check("condition_summary_present", len(condition_summary) > 0, f"{len(condition_summary)} rows")
    expected_regression_terms = 2 * frame["pde"].nunique()
    check(
        "tracking_regression_complete",
        len(regression) == expected_regression_terms,
        f"{len(regression)}/{expected_regression_terms} terms",
    )

    passed = all(item["passed"] for item in checks)
    report = {
        "status": "PASS" if passed else "FAIL",
        "results": str(results),
        "checks": checks,
        "scientific_readout": {
            "stencil_labels": assignment.groupby(["pde", "r"])["label"].apply(lambda x: int((x == "stencil").sum())).to_dict().__repr__(),
            "max_coefficient_error": float(frame["coefficient_max_abs"].max()),
            "max_stencil_mse": float(frame["mse_stencil"].max()),
        },
    }
    atomic_write_json(results / "validation_report.json", report)
    for item in checks:
        marker = "PASS" if item["passed"] else "FAIL"
        print(f"{marker:4s} {item['name']}: {item['detail']}")
    print(f"\nVALIDATION {report['status']}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
