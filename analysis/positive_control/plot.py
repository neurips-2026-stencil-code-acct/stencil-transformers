"""Build the compact S3 comparison and result-linked manuscript text."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STYLE = ROOT / "analysis" / "workshop_figures" / "neurips.mplstyle"
CONDITIONS = [
    ("heat", 0.10, "Heat .10"),
    ("heat", 0.25, "Heat .25"),
    ("heat", 0.40, "Heat .40"),
    ("lf", 0.10, "LF .10"),
    ("lf", 0.25, "LF .25"),
    ("lf", 0.40, "LF .40"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--positive", default="analysis/positive_control/results")
    parser.add_argument(
        "--ordinary-labels",
        default="analysis/what_do_heads_learn/results/heads.csv",
    )
    parser.add_argument(
        "--ordinary-statistics",
        default="analysis/parameter_sensitive/results/stats.csv",
    )
    parser.add_argument(
        "--ordinary-interventions",
        default="analysis/is_it_mechanistic/results/head_preserving/substitution.csv",
    )
    parser.add_argument(
        "--out",
        default="analysis/positive_control/results/figure_detectability",
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--width-inches", type=float, default=6.75)
    parser.add_argument("--height-inches", type=float, default=2.42)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def target_value(pde: str, r: float) -> float:
    return 1.0 - 2.0 * r if pde == "heat" else -r


def bootstrap_mean(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    draws = values[
        rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    ].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def tracking_rows(
    architecture: str,
    frame: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pde in ("heat", "lf"):
        part = frame[frame["pde"] == pde].copy()
        statistic = "centrality" if pde == "heat" else "asymmetry"
        target = "target_centrality" if pde == "heat" else "target_asymmetry"
        response = part.pivot(
            index="seed",
            columns="r",
            values=statistic,
        ).sort_index(axis=1)
        targets = (
            part.groupby("r")[target]
            .first()
            .reindex(response.columns)
            .to_numpy(dtype=np.float64)
        )
        if response.isna().any().any():
            raise RuntimeError(f"{architecture}/{pde}: incomplete seed matrix")
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
                    "architecture": architecture,
                    "pde": pde,
                    "statistic": statistic,
                    "term": term,
                    "coef": observed[term_index],
                    "ci_lo": np.quantile(draws[:, term_index], 0.025),
                    "ci_hi": np.quantile(draws[:, term_index], 0.975),
                    "n_seeds": len(seeds),
                }
            )
    return rows


def condition_summary(
    architecture: str,
    frame: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []
    for pde, r, label in CONDITIONS:
        part = frame[
            (frame["pde"] == pde) & np.isclose(frame["r"], r)
        ]
        statistic = "centrality" if pde == "heat" else "asymmetry"
        target = target_value(pde, r)
        mean, lo, hi = bootstrap_mean(
            part[statistic].to_numpy(),
            rng,
            n_bootstrap,
        )
        rows.append(
            {
                "architecture": architecture,
                "pde": pde,
                "r": r,
                "condition": label,
                "statistic": statistic,
                "target": target,
                "mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_seeds": len(part),
            }
        )
    return pd.DataFrame(rows)


def intervention_summary(
    positive: pd.DataFrame,
    ordinary_interventions: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    definitions = (
        ("Ord. fixed", "ordinary", "frozen_per_head_validation"),
        ("Ord. stencil", "ordinary", "analytical_stencil"),
        ("Control stencil", "control", "analytical_stencil"),
    )
    rows = []
    for pde, r, condition in CONDITIONS:
        for label, architecture, substitution in definitions:
            if architecture == "ordinary":
                part = ordinary_interventions[
                    (ordinary_interventions["pde"] == pde)
                    & np.isclose(ordinary_interventions["r"], r)
                    & (ordinary_interventions["substitution"] == substitution)
                ]
                values = part["excess"].to_numpy(dtype=np.float64)
            else:
                part = positive[
                    (positive["pde"] == pde) & np.isclose(positive["r"], r)
                ]
                values = part["excess_stencil"].to_numpy(dtype=np.float64)
            mean, lo, hi = bootstrap_mean(values, rng, n_bootstrap)
            rows.append(
                {
                    "pde": pde,
                    "r": r,
                    "condition": condition,
                    "intervention": label,
                    "architecture": architecture,
                    "substitution": substitution,
                    "mean_excess": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_seeds": len(values),
                }
            )
    return pd.DataFrame(rows)


def ordinary_seed_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["pde"].isin(["heat", "lf"])].copy()
    per_seed = (
        frame.groupby(["pde", "r", "seed"], as_index=False)[
            ["centrality", "asymmetry"]
        ]
        .mean()
        .sort_values(["pde", "r", "seed"])
    )
    per_seed["target_centrality"] = np.where(
        per_seed["pde"] == "heat",
        1.0 - 2.0 * per_seed["r"],
        0.0,
    )
    per_seed["target_asymmetry"] = np.where(
        per_seed["pde"] == "lf",
        -per_seed["r"],
        0.0,
    )
    return per_seed


def format_excess(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.0005:
        return "0.000"
    if magnitude < 0.1:
        return f"{value:.3f}"
    if magnitude < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def write_text(
    output_dir: Path,
    positive: pd.DataFrame,
    ordinary_labels_table: pd.DataFrame,
    tracking: pd.DataFrame,
    interventions: pd.DataFrame,
) -> None:
    positive_labels = int((positive["label"] == "stencil").sum())
    ordinary_labels = int((ordinary_labels_table["label"] == "stencil").sum())
    max_error = float(positive["coefficient_max_abs"].max())

    def slope(architecture: str, pde: str) -> pd.Series:
        return tracking[
            (tracking["architecture"] == architecture)
            & (tracking["pde"] == pde)
            & (tracking["term"] == "slope")
        ].iloc[0]

    ph, pl = slope("control", "heat"), slope("control", "lf")
    oh, ol = slope("ordinary", "heat"), slope("ordinary", "lf")
    ordinary_fixed = interventions[
        interventions["intervention"] == "Ord. fixed"
    ]["mean_excess"]
    ordinary_stencil = interventions[
        interventions["intervention"] == "Ord. stencil"
    ]["mean_excess"]
    control_stencil = interventions[
        interventions["intervention"] == "Control stencil"
    ]["mean_excess"]

    section = (
        "### Detectability control\n\n"
        f"With attention as the only spatial mixing operation, the unchanged "
        f"diagnostic assigned {positive_labels}/120 runs to the stencil and "
        f"recovered its coefficients (maximum absolute error "
        f"{max_error:.2e}). Tracking slopes were {ph.coef:.3f} for heat and "
        f"{pl.coef:.3f} for Lax--Friedrichs, versus {oh.coef:.3f} and "
        f"{ol.coef:.3f} in the trained transformers. Replacing learned "
        f"attention with the exact stencil changes the control result by "
        f"{control_stencil.min():.2e} to {control_stencil.max():.2e} on the "
        f"reported scale, compared with {ordinary_stencil.min():.2f} to "
        f"{ordinary_stencil.max():.2f} in the trained transformers. Fixing "
        f"each head at its validation-set mean gives {ordinary_fixed.min():.3f} to "
        f"{ordinary_fixed.max():.3f}.\n\n"
        "**Figure X. Detectability control.** (a) Stencil centrality or "
        "asymmetry versus attention centrality or asymmetry. Points are "
        "condition means and bars are 95% model-seed bootstrap intervals. "
        "(b) Increase in test MSE after fixing attention or replacing it with "
        "the exact stencil. The trained-transformer "
        f"The attention analysis assigned {ordinary_labels} heads to the stencil.\n"
    )
    (output_dir / "short_section.md").write_text(section, encoding="utf-8")

    tex = (
        "\\paragraph{Detectability control.} "
        f"With attention as the only spatial mixing operation, the unchanged "
        f"diagnostic assigned {positive_labels}/120 runs to the stencil and "
        f"recovered its coefficients (maximum absolute error "
        f"$ {max_error:.2e} $). Tracking slopes were {ph.coef:.3f} for heat "
        f"and {pl.coef:.3f} for Lax--Friedrichs, versus {oh.coef:.3f} and "
        f"{ol.coef:.3f} in the trained transformers. Replacing learned "
        f"attention with the exact stencil leaves the control predictions "
        f"effectively unchanged but not the trained-transformer predictions.\n"
    )
    (output_dir / "paper_ready.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    args = parse_args()
    positive_dir = (ROOT / args.positive).resolve()
    output = (ROOT / args.out).resolve()
    positive = pd.read_csv(positive_dir / "seed_metrics.csv")
    ordinary_labels_table = pd.read_csv((ROOT / args.ordinary_labels).resolve())
    ordinary_labels_table = ordinary_labels_table[
        ordinary_labels_table["pde"].isin(["heat", "lf"])
    ].copy()
    ordinary_statistics_table = pd.read_csv(
        (ROOT / args.ordinary_statistics).resolve()
    )
    ordinary_interventions_table = pd.read_csv(
        (ROOT / args.ordinary_interventions).resolve()
    )
    ordinary = ordinary_seed_statistics(ordinary_statistics_table)
    if len(positive) != 120 or len(ordinary) != 120:
        raise RuntimeError(
            f"Expected 120 seeds per architecture, got {len(positive)} and "
            f"{len(ordinary)}"
        )

    rng = np.random.default_rng(20260830)
    summaries = pd.concat(
        [
            condition_summary("control", positive, rng, args.bootstrap),
            condition_summary("ordinary", ordinary, rng, args.bootstrap),
        ],
        ignore_index=True,
    )
    tracking = pd.DataFrame(
        tracking_rows("control", positive, rng, args.bootstrap)
        + tracking_rows("ordinary", ordinary, rng, args.bootstrap)
    )
    interventions = intervention_summary(
        positive,
        ordinary_interventions_table,
        rng,
        args.bootstrap,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(output.parent / "figure1_condition_summary.csv", index=False)
    tracking.to_csv(output.parent / "comparison_tracking.csv", index=False)
    interventions.to_csv(
        output.parent / "figure1_intervention_summary.csv",
        index=False,
    )

    if not STYLE.exists():
        raise FileNotFoundError(STYLE)
    plt.style.use(STYLE)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(args.width_inches, args.height_inches),
        gridspec_kw={"width_ratios": [1.02, 1.18]},
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.23,
        top=0.84,
        wspace=0.31,
    )

    ax = axes[0]
    colors = {"heat": "#3274A1", "lf": "#E1812C"}
    markers = {"control": "o", "ordinary": "x"}
    labels_used: set[str] = set()
    for row in summaries.itertuples(index=False):
        key = (
            "Attention-only "
            if row.architecture == "control"
            else "Transformer "
        )
        key += "heat" if row.pde == "heat" else "LF"
        label = key if key not in labels_used else None
        labels_used.add(key)
        ax.errorbar(
            row.target,
            row.mean,
            yerr=[[row.mean - row.ci_lo], [row.ci_hi - row.mean]],
            color=colors[row.pde],
            marker=markers[row.architecture],
            markersize=4.5,
            markerfacecolor=(
                colors[row.pde] if row.architecture == "control" else "none"
            ),
            markeredgewidth=1.1,
            linestyle="none",
            capsize=1.8,
            elinewidth=0.8,
            label=label,
            zorder=3,
        )
    limits = (-0.48, 0.88)
    ax.plot(limits, limits, color="0.35", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(0, color="0.85", linewidth=0.5, zorder=0)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("Stencil centrality or asymmetry")
    ax.set_ylabel("Attention centrality or asymmetry")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left", ncol=2)

    ax = axes[1]
    intervention_labels = ["Ord. fixed", "Ord. stencil", "Control stencil"]
    values = np.full((len(CONDITIONS), len(intervention_labels)), np.nan)
    for row_index, (pde, r, _) in enumerate(CONDITIONS):
        for col_index, intervention in enumerate(intervention_labels):
            hit = interventions[
                (interventions["pde"] == pde)
                & np.isclose(interventions["r"], r)
                & (interventions["intervention"] == intervention)
            ]
            values[row_index, col_index] = float(hit["mean_excess"].iloc[0])
    transformed = np.sign(values) * np.log10(1.0 + np.abs(values))
    vmax = max(float(np.nanmax(np.abs(transformed))), 1e-12)
    ax.imshow(
        transformed,
        cmap="magma_r",
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_xticks(
        range(3),
        [
            "Head validation\nmean",
            "Transformer\nstencil",
            "Attention-only\nstencil",
        ],
    )
    ax.set_yticks(range(6), [item[2] for item in CONDITIONS])
    ax.tick_params(axis="both", length=0)
    ax.set_title("Increase in test MSE", pad=5)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            color = "white" if transformed[row, col] / vmax > 0.58 else "black"
            ax.text(
                col,
                row,
                format_excess(values[row, col]),
                ha="center",
                va="center",
                fontsize=6.4,
                color=color,
            )

    for ax, label in zip(axes, ("(a)", "(b)")):
        ax.text(
            -0.08,
            1.10,
            label,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=args.dpi)
    plt.close(fig)
    write_text(
        output.parent,
        positive,
        ordinary_labels_table,
        tracking,
        interventions,
    )
    print(output.with_suffix(".pdf"))
    print(output.with_suffix(".png"))
    print(output.parent / "short_section.md")


if __name__ == "__main__":
    main()
