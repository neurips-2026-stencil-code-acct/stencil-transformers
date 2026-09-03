"""Reusable helpers for publication-ready NeurIPS figures.

This module is opt-in. Existing plotting scripts retain their current
appearance unless they import and use this module explicitly.
"""

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


SINGLE_COL = 3.25
DOUBLE_COL = 6.75
DEFAULT_ASPECT = 0.62

COLORS = (
    "#d62728",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)

STYLE_PATH = Path(__file__).with_name("neurips.mplstyle")


def set_neurips_style():
    """Apply the NeurIPS stylesheet to subsequent Matplotlib figures."""
    plt.style.use(STYLE_PATH)


@contextmanager
def neurips_style():
    """Temporarily apply the NeurIPS stylesheet inside a ``with`` block."""
    with mpl.rc_context(fname=STYLE_PATH):
        yield


def figure_size(width="single", aspect=DEFAULT_ASPECT, rows=1):
    """Return a paper-sized ``(width, height)`` tuple in inches."""
    widths = {"single": SINGLE_COL, "double": DOUBLE_COL}
    if isinstance(width, str):
        try:
            width_inches = widths[width]
        except KeyError as exc:
            raise ValueError("width must be 'single', 'double', or inches") from exc
    else:
        width_inches = float(width)
    if width_inches <= 0 or aspect <= 0 or rows < 1:
        raise ValueError("width and aspect must be positive; rows must be >= 1")
    return width_inches, width_inches * float(aspect) * int(rows)


def subplots(nrows=1, ncols=1, width="single", aspect=DEFAULT_ASPECT, **kwargs):
    """Create a styled, paper-sized figure and axes."""
    set_neurips_style()
    kwargs.setdefault("figsize", figure_size(width, aspect=aspect, rows=nrows))
    return plt.subplots(nrows=nrows, ncols=ncols, **kwargs)


def label_panels(axes, labels=None, x=-0.16, y=1.05):
    """Add bold ``(a)``, ``(b)``, ... labels to one or more axes."""
    axes = list(mpl.cbook.flatten(axes))
    if labels is None:
        labels = [f"({chr(ord('a') + i)})" for i in range(len(axes))]
    if len(labels) != len(axes):
        raise ValueError("labels must contain one entry per axis")
    for ax, label in zip(axes, labels):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
        )


def save_figure(fig, path, **kwargs):
    """Save a figure with tight 300-dpi defaults and create its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("dpi", 300)
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", 0.02)
    fig.savefig(path, **kwargs)
