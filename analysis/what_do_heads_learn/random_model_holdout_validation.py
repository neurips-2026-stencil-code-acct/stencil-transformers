"""Validate attention labels on held-out random model initializations.

For each condition, one complete random model is held out.  All of that
model's heads are removed before entropy thresholds, predictor-specific
divergence scales, and fit thresholds are estimated from the other random
models.  The existing attention-assignment function is then applied to every head of
the held-out model.  This is repeated once per random model.

The model initialization is the independent held-out unit.  Head counts are
reported to show how labels arise within a model, but they are not treated as
independent replications.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from assign_heads import LABELS, assign


WORKSHOP_PDES = {"heat", "lf"}
WORKSHOP_PARAMETERS = {0.1, 0.25, 0.4}
SCIENTIFIC_LABELS = {"stencil", "acf", "spectral", "acf_or_spectral"}


def canonical_pde(value: str) -> str:
    name = str(value).strip().lower()
    if name in {"lf", "lax_friedrichs", "lax-friedrichs", "advection"}:
        return "lf"
    return name


def workshop_subset(table: pd.DataFrame, require_all_conditions: bool = True) -> pd.DataFrame:
    required = {
        "pde", "r", "seed", "layer", "head", "entropy",
        "d_stencil", "d_acf", "d_spectral", "spectral_kmax",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"baseline table is missing columns: {', '.join(missing)}")

    out = table.copy()
    out["pde"] = out["pde"].map(canonical_pde)
    out["r"] = pd.to_numeric(out["r"], errors="raise")
    out["seed"] = pd.to_numeric(out["seed"], errors="raise").astype(int)
    keep = out["pde"].isin(WORKSHOP_PDES) & out["r"].round(8).isin(
        WORKSHOP_PARAMETERS
    )
    out = out.loc[keep].copy()
    if out.empty:
        raise ValueError("no workshop conditions were found in the baseline table")

    if require_all_conditions:
        expected = {
            (pde, r) for pde in WORKSHOP_PDES for r in WORKSHOP_PARAMETERS
        }
        observed = set(zip(out["pde"], out["r"].round(8)))
        if observed != expected:
            raise ValueError(
                "baseline table does not contain exactly the six workshop "
                f"conditions; missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )
    return out.sort_values(["pde", "r", "seed", "layer", "head"]).reset_index(drop=True)


def unidentifiable_conditions(
    separation: pd.DataFrame,
    min_structure: float,
) -> tuple[tuple[str, float], ...]:
    required = {"condition", "d_acf_spectral"}
    missing = sorted(required - set(separation.columns))
    if missing:
        raise ValueError(
            f"predictor-separation table is missing columns: {', '.join(missing)}"
        )
    conditions = []
    for row in separation.itertuples(index=False):
        condition = str(row.condition)
        if "_r" not in condition:
            raise ValueError(f"cannot parse condition {condition!r}")
        pde, r_text = condition.rsplit("_r", 1)
        pde = canonical_pde(pde)
        r = float(r_text)
        if (
            pde in WORKSHOP_PDES
            and round(r, 8) in WORKSHOP_PARAMETERS
            and float(row.d_acf_spectral) < min_structure
        ):
            conditions.append((pde, r))
    return tuple(sorted(conditions))


def leave_one_random_model_out(
    baseline: pd.DataFrame,
    merge_unidentifiable: tuple[tuple[str, float], ...] = (),
    gate_pct: float = 5.0,
    floor_pct: float = 5.0,
    spectral_margin: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-head assignments and an exclusion audit for every model."""
    if not 0 <= gate_pct <= 100 or not 0 <= floor_pct <= 100:
        raise ValueError("gate and floor percentiles must lie between 0 and 100")
    if not 0 <= spectral_margin < 1:
        raise ValueError("spectral_margin must lie in [0, 1)")

    head_outputs = []
    audit_rows = []
    for (pde, r), condition in baseline.groupby(["pde", "r"], sort=True):
        seeds = sorted(condition["seed"].unique())
        if len(seeds) < 3:
            raise ValueError(f"{pde}_r{r:g} has only {len(seeds)} random models")
        heads_per_seed = condition.groupby("seed").size()
        if heads_per_seed.nunique() != 1:
            raise ValueError(
                f"{pde}_r{r:g} has inconsistent head counts across random models: "
                f"{heads_per_seed.to_dict()}"
            )

        for held_out_seed in seeds:
            held_out = condition[condition["seed"] == held_out_seed].copy()
            calibration = condition[condition["seed"] != held_out_seed].copy()
            calibration_seeds = sorted(calibration["seed"].unique())
            if held_out_seed in calibration_seeds:
                raise AssertionError("held-out seed remained in calibration data")
            if len(calibration_seeds) != len(seeds) - 1:
                raise AssertionError("calibration model count is incorrect")

            assigned = assign(
                held_out,
                calibration,
                gate_pct=gate_pct,
                floor_pct=floor_pct,
                spectral_margin=spectral_margin,
                merge_unidentifiable=merge_unidentifiable,
            ).reset_index(drop=True)
            assigned.insert(0, "held_out_seed", held_out_seed)
            assigned.insert(0, "validation_condition", f"{pde}_r{r:g}")
            assigned["structure_gate_pass"] = assigned["label"] != "ungated"
            assigned["assigned_scientific_label"] = assigned["label"].isin(
                SCIENTIFIC_LABELS
            )
            head_outputs.append(assigned)

            audit_rows.append({
                "pde": pde,
                "r": r,
                "condition": f"{pde}_r{r:g}",
                "held_out_seed": held_out_seed,
                "held_out_heads": len(held_out),
                "calibration_models": len(calibration_seeds),
                "calibration_heads": len(calibration),
                "held_out_seed_excluded": True,
                "gate_threshold": float(assigned["gate_threshold"].iloc[0]),
            })

    heads = pd.concat(head_outputs, ignore_index=True)
    audit = pd.DataFrame(audit_rows).sort_values(
        ["pde", "r", "held_out_seed"]
    ).reset_index(drop=True)
    return heads, audit


def summarize_models(heads: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouping = ["pde", "r", "held_out_seed"]
    for (pde, r, seed), group in heads.groupby(grouping, sort=True):
        labels = group["label"]
        row = {
            "pde": pde,
            "r": r,
            "condition": f"{pde}_r{r:g}",
            "held_out_seed": int(seed),
            "n_heads": len(group),
            "n_structure_gate_pass": int((labels != "ungated").sum()),
            "fraction_heads_structure_gate_pass": float((labels != "ungated").mean()),
            "any_structure_gate_pass": bool((labels != "ungated").any()),
            "n_assigned_scientific_label": int(labels.isin(SCIENTIFIC_LABELS).sum()),
            "fraction_heads_assigned_scientific_label": float(
                labels.isin(SCIENTIFIC_LABELS).mean()
            ),
            "any_assigned_scientific_label": bool(labels.isin(SCIENTIFIC_LABELS).any()),
            "n_no_fit": int((labels == "no_fit").sum()),
            "fraction_heads_no_fit": float((labels == "no_fit").mean()),
            "any_no_fit": bool((labels == "no_fit").any()),
        }
        for label in LABELS:
            row[f"n_label_{label}"] = int((labels == label).sum())
            row[f"fraction_heads_label_{label}"] = float((labels == label).mean())
            row[f"any_label_{label}"] = bool((labels == label).any())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["pde", "r", "held_out_seed"]
    ).reset_index(drop=True)


def _event_masks(group: pd.DataFrame) -> dict[str, pd.Series]:
    labels = group["label"]
    events = {
        "structure_gate_pass": labels != "ungated",
        "assigned_scientific_label": labels.isin(SCIENTIFIC_LABELS),
        "no_fit": labels == "no_fit",
    }
    for label in LABELS:
        events[f"label_{label}"] = labels == label
    return events


def summarize_rates(heads: pd.DataFrame) -> pd.DataFrame:
    """Rates use held-out models as units; head fractions remain descriptive."""
    groups = []
    for (pde, r), group in heads.groupby(["pde", "r"], sort=True):
        groups.append(("condition", pde, r, f"{pde}_r{r:g}", group))
    groups.append((
        "pooled_condition_evaluations",
        "all",
        np.nan,
        "all_workshop_conditions",
        heads,
    ))

    rows = []
    for scope, pde, r, condition, group in groups:
        model_keys = ["pde", "r", "held_out_seed"]
        n_models = group[model_keys].drop_duplicates().shape[0]
        for event, mask in _event_masks(group).items():
            event_rows = group.loc[mask, model_keys].drop_duplicates()
            rows.append({
                "scope": scope,
                "pde": pde,
                "r": r,
                "condition": condition,
                "event": event,
                "n_models": n_models,
                "n_models_with_at_least_one_event_head": len(event_rows),
                "model_rate": len(event_rows) / n_models,
                "n_heads": len(group),
                "n_event_heads": int(mask.sum()),
                "head_fraction_descriptive": float(mask.mean()),
            })
    return pd.DataFrame(rows)


def summarize_unique_seed_rates(heads: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated condition evaluations at the unique-seed level.

    The baseline generator reuses the same integer initialization seeds in all
    six workshop conditions. A seed is counted once here if an event occurs in
    any condition. Head-slot rates likewise count each ``(seed, layer, head)``
    slot once across conditions.
    """
    seed_key = ["held_out_seed"]
    slot_key = ["held_out_seed", "layer", "head"]
    n_unique_initializations = heads[seed_key].drop_duplicates().shape[0]
    n_unique_head_slots = heads[slot_key].drop_duplicates().shape[0]

    rows = []
    for event, mask in _event_masks(heads).items():
        event_heads = heads.loc[mask]
        n_initializations_with_event = event_heads[seed_key].drop_duplicates().shape[0]
        n_head_slots_with_event = event_heads[slot_key].drop_duplicates().shape[0]
        rows.append({
            "event": event,
            "n_unique_initializations": n_unique_initializations,
            "n_initializations_with_event": n_initializations_with_event,
            "initialization_rate": n_initializations_with_event / n_unique_initializations,
            "n_unique_head_slots": n_unique_head_slots,
            "n_head_slots_with_event_in_any_condition": n_head_slots_with_event,
            "head_slot_rate": n_head_slots_with_event / n_unique_head_slots,
            "n_condition_head_evaluations": len(heads),
            "n_event_condition_head_evaluations": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def wide_condition_summary(rates: pd.DataFrame) -> pd.DataFrame:
    reported_events = [
        "structure_gate_pass",
        "assigned_scientific_label",
        "no_fit",
        *[f"label_{label}" for label in LABELS],
    ]
    core = rates[rates["event"].isin(reported_events)]
    index = ["scope", "pde", "r", "condition", "n_models", "n_heads"]
    wide_model = core.pivot(index=index, columns="event", values="model_rate")
    wide_head = core.pivot(
        index=index, columns="event", values="head_fraction_descriptive"
    )
    wide_model.columns = [f"model_rate_any_{name}" for name in wide_model.columns]
    wide_head.columns = [f"head_fraction_{name}" for name in wide_head.columns]
    return wide_model.join(wide_head).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--baseline-heads",
        default="analysis/what_do_heads_learn/results/baseline_heads.csv",
    )
    parser.add_argument(
        "--predictor-separation",
        default="analysis/what_do_heads_learn/results/predictor_separation.csv",
    )
    parser.add_argument(
        "--out",
        default="analysis/what_do_heads_learn/results/robustness",
    )
    parser.add_argument("--gate-pct", type=float, default=5.0)
    parser.add_argument("--floor-pct", type=float, default=5.0)
    parser.add_argument("--spectral-margin", type=float, default=0.05)
    parser.add_argument("--min-structure", type=float, default=0.01)
    args = parser.parse_args()

    baseline = workshop_subset(pd.read_csv(args.baseline_heads))
    separation = pd.read_csv(args.predictor_separation)
    merged = unidentifiable_conditions(separation, args.min_structure)
    heads, audit = leave_one_random_model_out(
        baseline,
        merge_unidentifiable=merged,
        gate_pct=args.gate_pct,
        floor_pct=args.floor_pct,
        spectral_margin=args.spectral_margin,
    )
    models = summarize_models(heads)
    rates = summarize_rates(heads)
    unique_rates = summarize_unique_seed_rates(heads)
    summary = wide_condition_summary(rates)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    heads.to_csv(out_dir / "random_model_holdout_heads.csv", index=False)
    models.to_csv(out_dir / "random_model_holdout_models.csv", index=False)
    audit.to_csv(out_dir / "random_model_holdout_exclusion_audit.csv", index=False)
    rates.to_csv(out_dir / "random_model_holdout_event_rates.csv", index=False)
    unique_rates.to_csv(
        out_dir / "random_model_holdout_unique_seed_rates.csv", index=False
    )
    summary.to_csv(out_dir / "random_model_holdout_summary.csv", index=False)

    condition_seed_sets = [
        frozenset(group["seed"].unique())
        for _, group in baseline.groupby(["pde", "r"], sort=True)
    ]
    same_integer_seeds_reused = bool(
        condition_seed_sets
        and all(seed_set == condition_seed_sets[0] for seed_set in condition_seed_sets)
    )

    metadata = {
        "baseline_heads_input": str(Path(args.baseline_heads)),
        "predictor_separation_input": str(Path(args.predictor_separation)),
        "assignment_implementation": "analysis/what_do_heads_learn/assign_heads.py",
        "workshop_pdes": ["heat", "lf"],
        "workshop_parameters": [0.1, 0.25, 0.4],
        "gate_percentile": args.gate_pct,
        "floor_percentile": args.floor_pct,
        "spectral_margin": args.spectral_margin,
        "minimum_predictor_separation": args.min_structure,
        "merged_acf_spectral_conditions": [f"{p}_r{r:g}" for p, r in merged],
        "held_out_unit": "one condition-specific random model initialization",
        "same_integer_seeds_reused_across_conditions": same_integer_seeds_reused,
        "cross_condition_independence_note": (
            "The random-baseline generator resets the same integer seeds in "
            "every condition. Pooled condition rows are repeated evaluations "
            "of 30 shared weight initializations, not 180 independent "
            "initializations. The unique-seed summary counts each of the 30 "
            "initializations once if an event occurs in any condition."
        ),
        "calibration_rule": (
            "all heads of the held-out model are removed before thresholds "
            "and robust scales are estimated"
        ),
    }
    (out_dir / "random_model_holdout_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print("Random-model holdout validation")
    print(summary.to_string(index=False))
    print("\nMerged ACF/spectral conditions:", ", ".join(metadata[
        "merged_acf_spectral_conditions"
    ]) or "none")
    print(f"\nWrote holdout validation results to {out_dir}")


if __name__ == "__main__":
    main()
