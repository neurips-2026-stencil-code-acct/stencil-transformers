"""Run assign_heads and replace its mislabeled final console summary.

The canonical script's ``gated heads`` line counts only the three unmerged
hypothesis labels. It therefore excludes both ``acf_or_spectral`` (a valid
calibrated fit when those predictors are unidentifiable) and ``no_fit`` heads
that nevertheless passed the entropy gate. This reviewed wrapper preserves
all calculations and files, suppresses only that mislabeled line, and reports
the three distinct rates explicitly.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

import pandas as pd

import assign_heads


def output_directory(argv: list[str]) -> str:
    for index, token in enumerate(argv):
        if token == "--out" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--out="):
            return token.split("=", 1)[1]
    return "results"


def main() -> None:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        assign_heads.main()
    for line in captured.getvalue().splitlines():
        if not line.startswith("gated heads:"):
            print(line)

    heads_path = os.path.join(output_directory(sys.argv[1:]), "heads.csv")
    heads = pd.read_csv(heads_path)
    gate_pass = heads["label"] != "ungated"
    calibrated_fit = heads["label"].isin(
        ["stencil", "acf", "spectral", "acf_or_spectral"]
    )
    stencil = heads["label"] == "stencil"
    print(f"\nheads passing entropy gate: {gate_pass.mean():.1%} "
          f"({int(gate_pass.sum())}/{len(heads)})")
    print(f"heads passing baseline-calibrated fit: {calibrated_fit.mean():.1%} "
          f"({int(calibrated_fit.sum())}/{len(heads)})")
    print(f"heads labelled stencil: {stencil.mean():.1%} "
          f"({int(stencil.sum())}/{len(heads)})")


if __name__ == "__main__":
    main()
