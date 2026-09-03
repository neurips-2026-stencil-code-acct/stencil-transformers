"""Support-aware Fourier audit.

This stage consumes the frequency-resolved symbols written by
fourier_direction.py. The data generators excite only Fourier modes
k=1,...,5, and the linear finite-difference schemes preserve that support.
Accordingly, the primary operator-recovery scores use k=1,...,5.  Modes 6--8
are a near-support sensitivity check and modes 9--Nyquist test spectral
extrapolation.  A separate score weights k=1,...,5 by the empirical power of
the stored training inputs.

No checkpoint or training output is modified.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "what_do_heads_learn"))

import attention_common as C
from build_predictors import CONDITION_REGEX, find_conditions


TRAINING_MODE_MAX = 5
NEAR_SUPPORT_MODE_MAX = 8


def support_region(mode: int, nyquist: int) -> str:
    if mode == 0:
        return "dc"
    if mode <= min(TRAINING_MODE_MAX, nyquist):
        return "in_support_k1_5"
    if mode <= min(NEAR_SUPPORT_MODE_MAX, nyquist):
        return "near_support_k6_8"
    return "spectral_extrapolation_k9_nyquist"


def phase_abs_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (np.angle(a) - np.angle(b)))))


def selected_symbol_metrics(
    learned: np.ndarray,
    reference: np.ndarray,
    modes: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    """Compute complex-symbol errors over explicit non-DC modes."""
    learned = np.asarray(learned, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    modes = np.asarray(modes, dtype=int)
    if modes.ndim != 1 or len(modes) == 0:
        raise ValueError("at least one evaluation mode is required")
    if np.any(modes <= 0) or np.any(modes >= len(learned)):
        raise ValueError(f"modes must lie in [1, {len(learned) - 1}]")

    if weights is None:
        w = np.full(len(modes), 1.0 / len(modes), dtype=np.float64)
        weighting = "uniform"
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != modes.shape or np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite, non-negative, and match modes")
        if float(w.sum()) <= 0:
            raise ValueError("weights must have positive mass")
        w = w / w.sum()
        weighting = "empirical_training_power"

    z = learned[modes]
    target = reference[modes]
    err_ref = np.abs(z - target)
    err_id = np.abs(z - 1.0)
    rmse_ref = float(np.sqrt(np.sum(w * err_ref**2)))
    rmse_id = float(np.sqrt(np.sum(w * err_id**2)))
    denom = rmse_ref + rmse_id
    valid = (np.abs(target) > 1e-6) & (np.abs(z) > 1e-6)
    phase = phase_abs_error(z, target)

    return {
        "weighting": weighting,
        "n_modes": int(len(modes)),
        "mode_min": int(modes.min()),
        "mode_max": int(modes.max()),
        "rmse_reference": rmse_ref,
        "rmse_identity": rmse_id,
        "reference_advantage": float((rmse_id - rmse_ref) / denom) if denom else 0.0,
        "fraction_modes_closer_reference": float(np.sum(w * (err_ref < err_id))),
        "magnitude_rmse_reference": float(
            np.sqrt(np.sum(w * (np.abs(z) - np.abs(target)) ** 2))
        ),
        "phase_mae_reference": (
            float(np.sum(w[valid] * phase[valid]) / np.sum(w[valid]))
            if np.any(valid) else np.nan
        ),
    }


def evaluation_scopes(nyquist: int) -> list[tuple[str, np.ndarray]]:
    scopes = [
        ("in_support_k1_5", np.arange(1, min(TRAINING_MODE_MAX, nyquist) + 1)),
        (
            "near_support_k6_8",
            np.arange(TRAINING_MODE_MAX + 1, min(NEAR_SUPPORT_MODE_MAX, nyquist) + 1),
        ),
        (
            "spectral_extrapolation_k9_nyquist",
            np.arange(NEAR_SUPPORT_MODE_MAX + 1, nyquist + 1),
        ),
        ("all_non_dc_k1_nyquist", np.arange(1, nyquist + 1)),
    ]
    return [(name, modes) for name, modes in scopes if len(modes)]


def _condition_training_files(
    data_root: str,
    condition_regex: str,
    pde: str,
    r: float,
    files_per_condition: int,
) -> list[str]:
    candidates = [
        path
        for cpde, cr, path in find_conditions(data_root, condition_regex)
        if cpde == pde and np.isclose(cr, r)
    ]
    files: list[str] = []
    for cond_dir in candidates:
        flat = os.path.join(cond_dir, "train.npy")
        if os.path.exists(flat):
            files.append(flat)
        files.extend(sorted(glob.glob(os.path.join(cond_dir, "seed_*", "train.npy"))))
    files = sorted(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError(f"no train.npy found for {pde} r={r:g} under {data_root}")
    return files[:files_per_condition]


def _file_spectrum(
    path: str,
    max_trajectories: int,
    time_stride: int,
    chunk_trajectories: int,
) -> tuple[np.ndarray, int, int, int]:
    data = np.load(path, mmap_mode="r")
    if data.ndim != 3:
        raise ValueError(f"expected [trajectory,time,space] in {path}, got {data.shape}")
    n_trajectories = min(int(data.shape[0]), max_trajectories)
    n = int(data.shape[-1])
    power = np.zeros(n // 2 + 1, dtype=np.float64)
    observations = 0
    for start in range(0, n_trajectories, chunk_trajectories):
        stop = min(start + chunk_trajectories, n_trajectories)
        # Inputs to one-step training pairs exclude the final saved state.
        states = np.asarray(data[start:stop, :-1:time_stride, :], dtype=np.float64)
        fft = np.fft.rfft(states, axis=-1)
        power += np.sum(np.abs(fft) ** 2, axis=(0, 1))
        observations += states.shape[0] * states.shape[1]
    # Convert rFFT bins to their contribution to real-signal Parseval power.
    if n % 2 == 0 and len(power) > 2:
        power[1:-1] *= 2.0
    elif len(power) > 1:
        power[1:] *= 2.0
    return power, observations, n_trajectories, n


def empirical_training_spectra(
    conditions: pd.DataFrame,
    data_root: str,
    condition_regex: str,
    files_per_condition: int,
    max_trajectories: int,
    time_stride: int,
    chunk_trajectories: int,
) -> pd.DataFrame:
    rows = []
    for item in conditions.itertuples(index=False):
        paths = _condition_training_files(
            data_root, condition_regex, item.pde, float(item.r), files_per_condition
        )
        total = None
        total_observations = 0
        n_trajectories = 0
        n = None
        for path in paths:
            power, observations, n_traj, n_here = _file_spectrum(
                path, max_trajectories, time_stride, chunk_trajectories
            )
            if total is None:
                total = np.zeros_like(power)
                n = n_here
            if n_here != n:
                raise ValueError("all spectrum sources must use the same spatial grid")
            total += power
            total_observations += observations
            n_trajectories += n_traj
        assert total is not None and n is not None
        mean_power = total / max(total_observations, 1)
        non_dc_total = float(mean_power[1:].sum())
        support_total = float(mean_power[1 : TRAINING_MODE_MAX + 1].sum())
        support_fraction = support_total / non_dc_total if non_dc_total else np.nan
        source_paths = ";".join(os.path.relpath(p) for p in paths)
        for mode, value in enumerate(mean_power):
            rows.append(
                {
                    "pde": item.pde,
                    "r": float(item.r),
                    "condition": f"{item.pde}_r{float(item.r):g}",
                    "mode": mode,
                    "support_region": support_region(mode, n // 2),
                    "mean_power": float(value),
                    "non_dc_power_fraction": (
                        float(value / non_dc_total) if mode > 0 and non_dc_total else 0.0
                    ),
                    "support_power_fraction": support_fraction,
                    "n_observations": total_observations,
                    "n_trajectories": n_trajectories,
                    "time_stride": time_stride,
                    "source_files": source_paths,
                }
            )
        print(
            f"spectrum {item.pde} r={float(item.r):g}: "
            f"{support_fraction:.10f} of non-DC power in k=1..5 "
            f"from {len(paths)} file(s), {total_observations:,} states"
        )
    return pd.DataFrame(rows)


def analyse_support(symbols: pd.DataFrame, spectra: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["pde", "r", "condition", "seed", "input_step_index", "reference_scope"]
    for values, group in symbols.groupby(keys, sort=True):
        group = group.sort_values("mode")
        modes_saved = group["mode"].to_numpy(int)
        nyquist = int(modes_saved.max())
        if not np.array_equal(modes_saved, np.arange(nyquist + 1)):
            raise ValueError(f"incomplete mode sequence for {values}")
        learned = group["learned_real"].to_numpy() + 1j * group["learned_imag"].to_numpy()
        reference = group["reference_real"].to_numpy() + 1j * group["reference_imag"].to_numpy()
        base = dict(zip(keys, values))
        for scope, modes in evaluation_scopes(nyquist):
            row = dict(base)
            row["evaluation_scope"] = scope
            row.update(selected_symbol_metrics(learned, reference, modes))
            rows.append(row)

        spec = spectra[
            (spectra.pde == base["pde"]) & np.isclose(spectra.r, float(base["r"]))
        ].sort_values("mode")
        power = spec.set_index("mode")["mean_power"]
        train_modes = np.arange(1, min(TRAINING_MODE_MAX, nyquist) + 1)
        row = dict(base)
        row["evaluation_scope"] = "empirical_training_spectrum_k1_5"
        row.update(
            selected_symbol_metrics(
                learned,
                reference,
                train_modes,
                power.reindex(train_modes).to_numpy(float),
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_support(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["pde", "r", "reference_scope", "evaluation_scope", "weighting"]
    value_cols = [
        "rmse_reference",
        "rmse_identity",
        "reference_advantage",
        "fraction_modes_closer_reference",
        "magnitude_rmse_reference",
        "phase_mae_reference",
    ]
    rows = []
    for values, group in metrics.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        row["n_seeds"] = int(group.seed.nunique())
        row["n_modes"] = int(group.n_modes.iloc[0])
        row["mode_min"] = int(group.mode_min.iloc[0])
        row["mode_max"] = int(group.mode_max.iloc[0])
        for col in value_cols:
            x = group[col].dropna().to_numpy(float)
            row[f"{col}_mean"] = float(np.mean(x)) if len(x) else np.nan
            row[f"{col}_seed_p025"] = float(np.quantile(x, 0.025)) if len(x) else np.nan
            row[f"{col}_seed_p975"] = float(np.quantile(x, 0.975)) if len(x) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pde", "r", "evaluation_scope"])


def add_support_regions(symbols: pd.DataFrame) -> pd.DataFrame:
    out = symbols.copy()
    nyquist = int(out["mode"].max())
    out["support_region"] = [support_region(int(k), nyquist) for k in out["mode"]]
    return out


def plot_support_errors(summary: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scopes = [
        "in_support_k1_5",
        "near_support_k6_8",
        "spectral_extrapolation_k9_nyquist",
        "empirical_training_spectrum_k1_5",
    ]
    labels = ["in support\nk=1--5", "near\nk=6--8", "extrapolation\nk=9--32", "training-power\nweighted"]
    conditions = [(p, r) for p in ("heat", "lf", "wave")
                  for r in sorted(summary.loc[summary.pde == p, "r"].unique())]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharey=False)
    x = np.arange(len(scopes))
    width = 0.36
    for ax, (pde, r) in zip(axes.flat, conditions):
        frame = summary[(summary.pde == pde) & np.isclose(summary.r, r)].set_index(
            "evaluation_scope"
        )
        ref = [frame.loc[s, "rmse_reference_mean"] for s in scopes]
        identity = [frame.loc[s, "rmse_identity_mean"] for s in scopes]
        ax.bar(x - width / 2, ref, width, color="#E45756", label="PDE reference")
        ax.bar(x + width / 2, identity, width, color="#4C78A8", label="identity")
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_yscale("log")
        ax.set_title(f"{pde}, r={r:g}")
        ax.grid(axis="y", alpha=0.2)
        if pde == "wave":
            ax.text(0.02, 0.96, "spatial part only", transform=ax.transAxes,
                    va="top", fontsize=8, color="#555555")
    for ax in axes[:, 0]:
        ax.set_ylabel("complex-symbol RMSE (log scale)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("In-support recovery and spectral extrapolation", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--symbols", required=True, help="fourier_symbols.csv")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition-regex", default=CONDITION_REGEX)
    ap.add_argument("--spectrum-files-per-condition", type=int, default=1)
    ap.add_argument("--spectrum-max-trajectories", type=int, default=800)
    ap.add_argument("--spectrum-time-stride", type=int, default=1)
    ap.add_argument("--spectrum-chunk-trajectories", type=int, default=16)
    ap.add_argument("--min-support-power-fraction", type=float, default=0.999999)
    args = ap.parse_args()

    if args.spectrum_files_per_condition < 1:
        raise ValueError("--spectrum-files-per-condition must be positive")
    if args.spectrum_time_stride < 1 or args.spectrum_chunk_trajectories < 1:
        raise ValueError("spectrum stride and chunk size must be positive")

    os.makedirs(args.out, exist_ok=True)
    symbols = pd.read_csv(args.symbols)
    required = {
        "pde", "r", "condition", "seed", "input_step_index", "reference_scope",
        "mode", "learned_real", "learned_imag", "reference_real", "reference_imag",
    }
    missing = required.difference(symbols.columns)
    if missing:
        raise ValueError(f"symbols file is missing columns: {sorted(missing)}")

    conditions = symbols[["pde", "r"]].drop_duplicates().sort_values(["pde", "r"])
    spectra = empirical_training_spectra(
        conditions,
        args.data_root,
        args.condition_regex,
        args.spectrum_files_per_condition,
        args.spectrum_max_trajectories,
        args.spectrum_time_stride,
        args.spectrum_chunk_trajectories,
    )
    min_fraction = float(spectra.groupby(["pde", "r"]).support_power_fraction.first().min())
    if min_fraction < args.min_support_power_fraction:
        raise AssertionError(
            f"minimum k=1..5 power fraction {min_fraction:.10f} is below "
            f"{args.min_support_power_fraction}; check the stated training support"
        )

    symbols = add_support_regions(symbols)
    metrics = analyse_support(symbols, spectra)
    summary = summarise_support(metrics)

    symbols.to_csv(os.path.join(args.out, "fourier_symbols_support_aware.csv"), index=False)
    spectra.to_csv(os.path.join(args.out, "training_spectrum.csv"), index=False)
    metrics.to_csv(os.path.join(args.out, "fourier_support_metrics.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "fourier_support_summary.csv"), index=False)
    plot_support_errors(summary, os.path.join(args.out, "support_errors.png"))

    show_scopes = [
        "in_support_k1_5",
        "empirical_training_spectrum_k1_5",
        "spectral_extrapolation_k9_nyquist",
        "all_non_dc_k1_nyquist",
    ]
    show = summary[summary.evaluation_scope.isin(show_scopes)][[
        "pde", "r", "reference_scope", "evaluation_scope", "n_seeds",
        "rmse_reference_mean", "rmse_identity_mean", "reference_advantage_mean",
        "fraction_modes_closer_reference_mean",
    ]]
    print("\nsupport-aware Jacobian symbol summary")
    print(show.round(5).to_string(index=False))
    print(
        "\nPrimary recovery scope: k=1..5.  k=9..Nyquist is an "
        "out-of-support spectral extrapolation test."
    )
    print("Wave comparisons remain spatial-part-only, not full propagator tests.")
    print(f"\nwrote support-aware Fourier outputs to {args.out}")


if __name__ == "__main__":
    main()
