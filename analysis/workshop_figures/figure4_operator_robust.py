"""Corrected Figure 4: effective operator and support-aware Fourier audit."""

from pathlib import Path
import hashlib
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neurips_figure_style import COLORS, figure_size, save_figure, set_neurips_style
from workshop_common import (
    PDE_COLORS,
    PDE_MARKERS,
    condition_order,
    heat_lf,
    result_path,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output_robustness"
PROFILE_KEY = re.compile(
    r"^(?P<pde>heat|lf)_r(?P<r>[0-9.]+)_seed(?P<seed>[0-9]+)_step(?P<step>[0-9]+)$"
)


def panel_label(ax, label, x=-0.18, y=1.10):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )


def profile_stack(store, pde, r):
    rows = []
    for key in store.files:
        match = PROFILE_KEY.match(key)
        if not match:
            continue
        if (
            match.group("pde") == pde
            and np.isclose(float(match.group("r")), r)
            and int(match.group("step")) == 0
        ):
            rows.append((int(match.group("seed")), np.asarray(store[key], dtype=float)))
    if not rows:
        raise ValueError(f"no Jacobian profiles for {pde} r={r:g}")
    rows.sort(key=lambda item: item[0])
    if len(rows) != 20:
        raise ValueError(f"expected 20 Jacobian profiles for {pde} r={r:g}")
    return np.stack([profile for _, profile in rows])


def bootstrap_profile(values, label, n_bootstrap=20_000):
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    rng = np.random.default_rng(
        np.random.SeedSequence([20_260_824, int.from_bytes(digest[:8], "little")])
    )
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975], axis=0)
    return values.mean(axis=0), lower, upper


def stencil_profile(pde, r, offsets):
    out = np.zeros_like(offsets, dtype=float)
    if pde == "heat":
        weights = {-1: r, 0: 1 - 2 * r, 1: r}
    else:
        weights = {-1: (1 + r) / 2, 0: 0.0, 1: (1 - r) / 2}
    for offset, weight in weights.items():
        out[offsets == offset] = weight
    return out


def draw_profile(ax, store, pde, r, show_ylabel=True):
    offsets = np.asarray(store["offsets"], dtype=int)
    profiles = profile_stack(store, pde, r)
    keep = np.abs(offsets) <= 4
    order = np.argsort(offsets[keep])
    x = offsets[keep][order]
    values = profiles[:, keep][:, order]
    mean, lower, upper = bootstrap_profile(values, f"{pde}|{r:g}|profile")
    target = stencil_profile(pde, r, x)
    identity = (x == 0).astype(float)
    color = PDE_COLORS[pde]

    ax.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, mean, color=color, marker="o", label="Model Jacobian")
    ax.plot(x, target, color="#222222", linestyle="--", marker="x", label="PDE")
    ax.plot(x, identity, color="#888888", linestyle=":", label="Identity")
    ax.axhline(0, color="#bbbbbb", linewidth=0.5)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xlabel("Spatial offset $d$")
    if show_ylabel:
        ax.set_ylabel("Jacobian coefficient")
    symbol = "r" if pde == "heat" else r"\nu"
    ax.set_title(f"{'Heat' if pde == 'heat' else 'LF'} ${symbol}={r:g}$")
    ax.set_ylim(-0.08, 1.06)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.4)


def draw_update_residual(ax, path):
    data = heat_lf(pd.read_csv(path))
    for x, (pde, r) in enumerate(condition_order()):
        row = data[(data["pde"] == pde) & np.isclose(data["r"], r)]
        if len(row) != 1:
            raise ValueError(f"missing J-I summary for {pde} {r:g}")
        row = row.iloc[0]
        marker = PDE_MARKERS[pde]
        ax.errorbar(
            x - 0.10,
            row["circulant_residual_j_mean"],
            yerr=[[
                row["circulant_residual_j_mean"]
                - row["circulant_residual_j_ci_lower"]
            ], [
                row["circulant_residual_j_ci_upper"]
                - row["circulant_residual_j_mean"]
            ]],
            color="#777777",
            marker=marker,
            markerfacecolor="white",
            capsize=2,
            linestyle="none",
        )
        ax.errorbar(
            x + 0.10,
            row["circulant_residual_j_minus_i_mean"],
            yerr=[[
                row["circulant_residual_j_minus_i_mean"]
                - row["circulant_residual_j_minus_i_ci_lower"]
            ], [
                row["circulant_residual_j_minus_i_ci_upper"]
                - row["circulant_residual_j_minus_i_mean"]
            ]],
            color=PDE_COLORS[pde],
            marker=marker,
            capsize=2,
            linestyle="none",
        )

    ax.set_yscale("log")
    ax.set_ylim(2e-4, 0.15)
    ax.set_xticks(
        range(6),
        [
            f"{'H' if pde == 'heat' else 'LF'}\n{r:g}"
            for pde, r in condition_order()
        ],
    )
    ax.set_ylabel("Circulant residual")
    ax.set_title(r"Circulant residual of $J$ and $J-I$")
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", which="major", color="#e0e0e0", linewidth=0.4)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], color="#777777", marker="o", markerfacecolor="white",
                linestyle="none", label="$J$"
            ),
            Line2D(
                [0], [0], color="#222222", marker="o", linestyle="none",
                label="$J-I$"
            ),
        ],
        loc="lower right",
    )


def draw_spectrum(ax, path):
    data = heat_lf(pd.read_csv(path))
    data = data[(data["mode"] >= 1) & (data["mode"] <= 32)]
    summary = data.groupby(["pde", "mode"], as_index=False)[
        "non_dc_power_fraction"
    ].mean()
    ax.axvspan(0.5, 5.5, color=COLORS[3], alpha=0.12, linewidth=0)
    ax.axvspan(5.5, 8.5, color=COLORS[2], alpha=0.10, linewidth=0)
    ax.axvspan(8.5, 32.5, color="#bdbdbd", alpha=0.10, linewidth=0)
    for pde in ("heat", "lf"):
        group = summary[summary["pde"] == pde]
        ax.plot(
            group["mode"],
            np.maximum(group["non_dc_power_fraction"], 1e-12),
            color=PDE_COLORS[pde],
            marker=PDE_MARKERS[pde],
            markevery=[0, 1, 2, 3, 4],
            label="Heat" if pde == "heat" else "LF",
        )
    ax.set_yscale("log")
    ax.set_xlim(0.5, 32.5)
    ax.set_ylim(1e-12, 1)
    ax.set_xticks([1, 5, 8, 16, 24, 32])
    ax.set_xlabel("Fourier mode $k$")
    ax.set_ylabel("Fraction of spatial variance")
    ax.set_title("Frequencies in generated states")
    ax.legend(loc="upper right")
    ax.grid(axis="y", which="major", color="#e0e0e0", linewidth=0.4)


def delta_text(value):
    value = float(value)
    if abs(value) < 0.01:
        return f"{value:+.4f}"
    if abs(value) < 0.1:
        return f"{value:+.3f}"
    return f"{value:+.2f}"


def draw_fourier_table(ax, path):
    data = heat_lf(pd.read_csv(path))
    scopes = [
        "k1_5_empirical_power",
        "k1_5_uniform",
        "k6_8_uniform",
        "k9_32_uniform",
    ]
    labels = [
        "Data weighted\n$k=1$-$5$",
        "Equal weights\n$k=1$-$5$",
        "$k=6$-$8$",
        "$k=9$-$32$",
    ]
    matrix = np.zeros((6, 4), dtype=int)
    annotations = np.empty((6, 4), dtype=object)
    for i, (pde, r) in enumerate(condition_order()):
        for j, scope in enumerate(scopes):
            row = data[
                (data["pde"] == pde)
                & np.isclose(data["r"], r)
                & (data["scope_label"] == scope)
            ]
            if len(row) != 1:
                raise ValueError(f"missing Fourier summary for {pde} {r:g}, {scope}")
            row = row.iloc[0]
            lower = float(row["mean_delta_ci_lower"])
            upper = float(row["mean_delta_ci_upper"])
            if lower > 0:
                matrix[i, j] = 1
            elif upper < 0:
                matrix[i, j] = -1
            else:
                matrix[i, j] = 0
            annotations[i, j] = delta_text(
                row["mean_delta_identity_minus_pde"]
            )

    cmap = ListedColormap([PDE_COLORS["lf"], "#f2f2f2", PDE_COLORS["heat"]])
    ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] else "black"
            ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=6.5, color=color)
    ax.set_xticks(range(4), labels)
    ax.set_yticks(
        range(6),
        [f"{'Heat' if pde == 'heat' else 'LF'} {r:.2f}" for pde, r in condition_order()],
    )
    ax.tick_params(which="both", bottom=False, left=False)
    ax.set_title("Fourier response comparison")
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    set_neurips_style()
    fig, ax_fourier = plt.subplots(
        figsize=figure_size(5.0, aspect=0.52),
    )
    draw_fourier_table(
        ax_fourier,
        result_path(
            "analysis",
            "robustness_checks",
            "results",
            "paired_bootstrap",
            "fourier_bootstrap_summary.csv",
        ),
    )
    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.18, right=0.98)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        save_figure(fig, OUTPUT_DIR / f"figure4_operator_robust.{suffix}")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR / 'figure4_operator_robust.pdf'}")
    print(f"wrote {OUTPUT_DIR / 'figure4_operator_robust.png'}")


if __name__ == "__main__":
    main()
