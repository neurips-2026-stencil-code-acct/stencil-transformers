"""Shared data and plotting utilities for the workshop figures."""

from pathlib import Path

import numpy as np
import pandas as pd

from neurips_figure_style import COLORS, save_figure


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
OUTPUT_DIR = HERE / "output"
PDES = ("heat", "lf")
PARAMETERS = (0.1, 0.25, 0.4)
PDE_COLORS = {"heat": COLORS[0], "lf": COLORS[1]}
PDE_MARKERS = {"heat": "o", "lf": "s"}


def result_path(*parts):
    """Resolve a path under the repository and fail with a useful message."""
    path = REPO_ROOT.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"required analysis artifact not found: {path}")
    return path


def heat_lf(frame):
    """Return the workshop PDE subset with normalized PDE names."""
    out = frame.copy()
    out["pde"] = out["pde"].replace({"lax_friedrichs": "lf"})
    return out[out["pde"].isin(PDES)].copy()


def condition_order():
    return [(pde, r) for pde in PDES for r in PARAMETERS]


def condition_label(pde, r):
    symbol = "r" if pde == "heat" else r"$\nu$"
    name = "Heat" if pde == "heat" else "LF"
    return f"{name}\n{symbol}={r:g}"


def deterministic_jitter(n, width=0.12):
    """Return centered deterministic jitter, avoiding random plot changes."""
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-width, width, n)


def percentile_interval(values):
    values = np.asarray(values, dtype=float)
    return np.nanmean(values), *np.nanpercentile(values, [2.5, 97.5])


def mean_with_interval(ax, x, values, color, marker="o", zorder=4):
    """Draw a mean and a 2.5th--97.5th percentile interval."""
    mean, low, high = percentile_interval(values)
    ax.errorbar(
        x,
        mean,
        yerr=[[mean - low], [high - mean]],
        color=color,
        marker=marker,
        markeredgecolor="white",
        markeredgewidth=0.45,
        capsize=2,
        linestyle="none",
        zorder=zorder,
    )
    return mean, low, high


def save_pdf_png(fig, stem):
    """Save both vector PDF and high-resolution PNG outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    png = OUTPUT_DIR / f"{stem}.png"
    save_figure(fig, pdf)
    save_figure(fig, png)
    return pdf, png
