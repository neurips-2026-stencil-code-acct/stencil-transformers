"""Deterministic tests for the workshop robustness post-processing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from random_model_holdout_validation import (
    leave_one_random_model_out,
    summarize_models,
    summarize_rates,
    summarize_unique_seed_rates,
)
from multiple_testing import bh_adjust, recompute_workshop_bh


def test_bh_known_values() -> None:
    observed = bh_adjust([0.01, 0.04, 0.03, 0.002])
    expected = np.array([0.02, 0.04, 0.04, 0.008])
    np.testing.assert_allclose(observed, expected, atol=1e-15, rtol=0)


def test_workshop_filter_and_per_metric_families() -> None:
    rows = []
    conditions = [
        ("heat", 0.1), ("heat", 0.25), ("heat", 0.4),
        ("lf", 0.1), ("lf", 0.25), ("lf", 0.4),
        ("wave", 0.1),
    ]
    for metric, scale in [("entropy", 1.0), ("locality", 0.5)]:
        for index, (pde, r) in enumerate(conditions, start=1):
            rows.append({
                "pde": pde,
                "r": r,
                "condition": f"{pde}_r{r:g}",
                "metric": metric,
                "p_value": scale * index / 100,
                "p_value_bh": 0.99,
                "significant_bh_0.05": False,
            })
    result = recompute_workshop_bh(pd.DataFrame(rows))
    assert len(result) == 12
    assert set(result["pde"]) == {"heat", "lf"}
    assert (result.groupby("metric").size() == 6).all()
    assert (result["bh_family_size"] == 6).all()


def synthetic_baseline() -> pd.DataFrame:
    rows = []
    # Four model seeds, with three heads each.  Seed 3 is deliberately extreme
    # so its gate threshold changes when that entire model is held out.
    for seed in range(4):
        for head in range(3):
            entropy = 0.80 + 0.02 * seed + 0.003 * head
            if seed == 3:
                entropy = 0.20 + 0.003 * head
            base = 0.10 + 0.02 * seed + 0.005 * head
            rows.append({
                "pde": "heat", "r": 0.1, "seed": seed,
                "layer": 0, "head": head, "is_baseline": True,
                "entropy": entropy, "peak": 0.1, "window_mass": 0.2,
                "d_stencil": base,
                "d_acf": base + 0.03,
                "d_spectral": base + 0.06,
                "spectral_kmax": 5,
            })
    return pd.DataFrame(rows)


def test_whole_model_exclusion_and_summaries() -> None:
    baseline = synthetic_baseline()
    heads, audit = leave_one_random_model_out(
        baseline,
        gate_pct=5.0,
        floor_pct=5.0,
        spectral_margin=0.05,
    )
    assert len(heads) == len(baseline)
    assert len(audit) == baseline["seed"].nunique()
    assert audit["held_out_seed_excluded"].all()
    assert (audit["held_out_heads"] == 3).all()
    assert (audit["calibration_models"] == 3).all()
    assert (audit["calibration_heads"] == 9).all()

    for row in audit.itertuples(index=False):
        expected_gate = np.percentile(
            baseline.loc[baseline["seed"] != row.held_out_seed, "entropy"], 5.0
        )
        assert np.isclose(row.gate_threshold, expected_gate)

    models = summarize_models(heads)
    assert len(models) == 4
    assert (models["n_heads"] == 3).all()
    rates = summarize_rates(heads)
    assert set(rates["scope"]) == {
        "condition", "pooled_condition_evaluations"
    }
    assert {
        "structure_gate_pass", "assigned_scientific_label", "no_fit",
        "label_stencil", "label_acf", "label_spectral",
    }.issubset(set(rates["event"]))
    unique_rates = summarize_unique_seed_rates(heads)
    assert (unique_rates["n_unique_initializations"] == 4).all()
    assert (unique_rates["n_unique_head_slots"] == 12).all()
    assert (
        unique_rates["n_condition_head_evaluations"] == len(heads)
    ).all()


def main() -> None:
    tests = [
        test_bh_known_values,
        test_workshop_filter_and_per_metric_families,
        test_whole_model_exclusion_and_summaries,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} deterministic attention robustness tests passed")


if __name__ == "__main__":
    main()
