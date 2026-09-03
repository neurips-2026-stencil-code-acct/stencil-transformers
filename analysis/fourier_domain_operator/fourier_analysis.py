"""Fourier-domain audit of the effective learned operator.

The primary analysis transforms the diagonal-averaged Jacobian rows into
Fourier symbols.  This is deliberately an input-output analysis: an attention
profile spectrum is not the model transfer function because value/output
projections, MLPs, residual paths, and later layers intervene.

Optionally, finite-amplitude sinusoidal probes can be run on checkpoints.  The
probe response is measured after subtracting the model output on a constant
background, and is reported separately from the local Jacobian symbol.

The wave reference stored by the attention analysis is only the spatial part of a two-time-level
leapfrog update.  Wave rows are therefore marked ``spatial_part_only`` and
must not be read as a complete wave-propagator comparison.
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
sys.path.insert(0, os.path.join(HERE, "..", "is_it_mechanistic"))

import attention_common as C
from build_predictors import CONDITION_REGEX
from extract_attention_profiles import condition_data_dir
from mechanistic_model import load_eval_pairs, load_model, model_output, to_batch


PROFILE_RE = re.compile(
    r"^(?P<pde>heat|lf|wave)_r(?P<r>[0-9.]+)_seed(?P<seed>\d+)_step(?P<step>\d+)$"
)


def embed_offsets(values: np.ndarray, offsets: np.ndarray, n: int) -> np.ndarray:
    """Embed signed-offset values in a row indexed modulo N."""
    row = np.zeros(n, dtype=np.float64)
    row[np.asarray(offsets, dtype=int) % n] = np.asarray(values, dtype=np.float64)
    return row


def symbol_from_offsets(values: np.ndarray, offsets: np.ndarray, n: int) -> np.ndarray:
    """Return the non-negative-frequency symbol using the shared FFT convention."""
    return np.fft.fft(embed_offsets(values, offsets, n))[: n // 2 + 1]


def reference_symbol(pred, condition: str, n: int) -> np.ndarray:
    w = np.asarray(pred[f"{condition}/stencil_raw"], dtype=np.float64)
    d = np.asarray(pred[f"{condition}/stencil_offsets"], dtype=int)
    return symbol_from_offsets(w, d, n)


def reference_scope(pde: str) -> str:
    return "spatial_part_only" if pde == "wave" else "full_one_step"


def frequency_band(q: float) -> str:
    if q <= 0.25:
        return "low"
    if q <= 0.50:
        return "mid"
    return "high"


def _phase_abs_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (np.angle(a) - np.angle(b)))))


def symbol_metrics(learned: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Metrics over non-DC frequencies; DC gain is reported separately."""
    learned = np.asarray(learned, dtype=np.complex128)
    ref = np.asarray(ref, dtype=np.complex128)
    identity = np.ones_like(ref)
    sl = slice(1, None)
    err_ref = np.abs(learned[sl] - ref[sl])
    err_id = np.abs(learned[sl] - identity[sl])
    rmse_ref = float(np.sqrt(np.mean(err_ref ** 2)))
    rmse_id = float(np.sqrt(np.mean(err_id ** 2)))
    denom = rmse_ref + rmse_id
    out = {
        "dc_gain_real": float(learned[0].real),
        "dc_gain_imag": float(learned[0].imag),
        "rmse_reference": rmse_ref,
        "rmse_identity": rmse_id,
        "reference_advantage": float((rmse_id - rmse_ref) / denom) if denom else 0.0,
        "fraction_modes_closer_reference": float(np.mean(err_ref < err_id)),
        "magnitude_rmse_reference": float(
            np.sqrt(np.mean((np.abs(learned[sl]) - np.abs(ref[sl])) ** 2))
        ),
    }
    valid = (np.abs(ref[sl]) > 1e-6) & (np.abs(learned[sl]) > 1e-6)
    out["phase_mae_reference"] = (
        float(np.mean(_phase_abs_error(learned[sl][valid], ref[sl][valid])))
        if np.any(valid) else np.nan
    )
    m = len(ref) - 1
    q = np.arange(1, m + 1) / max(m, 1)
    for band in ("low", "mid", "high"):
        keep = np.array([frequency_band(x) == band for x in q])
        out[f"rmse_reference_{band}"] = float(np.sqrt(np.mean(err_ref[keep] ** 2)))
        out[f"rmse_identity_{band}"] = float(np.sqrt(np.mean(err_id[keep] ** 2)))
    return out


def analyse_jacobians(path: str, predictors: str):
    archive = np.load(path)
    pred = np.load(predictors)
    offsets = np.asarray(archive["offsets"], dtype=int)
    n = len(offsets)
    rows, points = [], []
    for name in sorted(k for k in archive.files if k != "offsets"):
        match = PROFILE_RE.match(name)
        if not match:
            raise ValueError(f"unrecognised Jacobian profile key: {name}")
        pde = match.group("pde")
        r = float(match.group("r"))
        seed = int(match.group("seed"))
        step = int(match.group("step"))
        condition = f"{pde}_r{r:g}"
        learned = symbol_from_offsets(archive[name], offsets, n)
        ref = reference_symbol(pred, condition, n)
        metrics = symbol_metrics(learned, ref)
        metrics.update({
            "pde": pde, "r": r, "condition": condition, "seed": seed,
            "input_step_index": step, "reference_scope": reference_scope(pde),
        })
        rows.append(metrics)
        for k, (z, target) in enumerate(zip(learned, ref)):
            q = k / (n // 2)
            points.append({
                "pde": pde, "r": r, "condition": condition, "seed": seed,
                "input_step_index": step, "reference_scope": reference_scope(pde),
                "mode": k, "normalized_frequency": q,
                "band": "dc" if k == 0 else frequency_band(q),
                "learned_real": z.real, "learned_imag": z.imag,
                "learned_magnitude": abs(z), "learned_phase": np.angle(z),
                "reference_real": target.real, "reference_imag": target.imag,
                "reference_magnitude": abs(target), "reference_phase": np.angle(target),
                "identity_distance": abs(z - 1.0),
                "reference_distance": abs(z - target),
            })
    return pd.DataFrame(rows), pd.DataFrame(points)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dc_gain_real", "rmse_reference", "rmse_identity", "reference_advantage",
        "fraction_modes_closer_reference", "magnitude_rmse_reference",
        "phase_mae_reference", "rmse_reference_low", "rmse_reference_mid",
        "rmse_reference_high",
    ]
    records = []
    for (pde, r, scope), group in metrics.groupby(["pde", "r", "reference_scope"]):
        rec = {"pde": pde, "r": r, "reference_scope": scope, "n_seeds": len(group)}
        for col in cols:
            v = group[col].dropna().to_numpy(float)
            rec[f"{col}_mean"] = float(np.mean(v)) if len(v) else np.nan
            rec[f"{col}_seed_p025"] = float(np.quantile(v, 0.025)) if len(v) else np.nan
            rec[f"{col}_seed_p975"] = float(np.quantile(v, 0.975)) if len(v) else np.nan
        records.append(rec)
    return pd.DataFrame(records).sort_values(["pde", "r"])


def plot_symbols(points: pd.DataFrame, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [(p, r) for p in ("heat", "lf", "wave")
                  for r in sorted(points.loc[points.pde == p, "r"].unique())]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    for ax, (pde, r) in zip(axes.flat, conditions):
        g = points[(points.pde == pde) & (points.r == r)]
        pivot = g.pivot(index="seed", columns="mode", values="learned_magnitude")
        x = g.groupby("mode")["normalized_frequency"].first().to_numpy()
        ref = g.groupby("mode")["reference_magnitude"].first().to_numpy()
        med = pivot.median(axis=0).to_numpy()
        lo = pivot.quantile(0.025, axis=0).to_numpy()
        hi = pivot.quantile(0.975, axis=0).to_numpy()
        ax.fill_between(x, lo, hi, color="#4C78A8", alpha=0.20, label="seed 2.5–97.5%")
        ax.plot(x, med, color="#4C78A8", lw=2, label="learned median")
        ax.plot(x, ref, color="#E45756", lw=2, ls="--", label="reference")
        ax.axhline(1.0, color="black", lw=1, ls=":", label="identity")
        ax.set_title(f"{pde}, r={r:g}")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("normalized spatial frequency")
    for ax in axes[:, 0]:
        ax.set_ylabel("symbol magnitude")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Learned Jacobian Fourier symbols", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _parse_csv_numbers(value: str, cast=float):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def probe_checkpoints(args, pred):
    import torch

    paths = sorted(glob.glob(args.runs, recursive=True))
    if not paths:
        raise SystemExit(f"no checkpoints matched {args.runs}")
    modes = _parse_csv_numbers(args.probe_modes, int)
    amplitudes = _parse_csv_numbers(args.probe_amplitudes, float)
    phases = np.linspace(0.0, 2 * np.pi, args.n_phases, endpoint=False)
    records = []
    for index, path in enumerate(paths, 1):
        meta = C.parse_run(path, args.run_regex)
        if meta is None:
            continue
        condition = f"{meta.pde}_r{meta.r:g}"
        ddir = condition_data_dir(args.data_root, meta, args.data_template,
                                  args.condition_regex)
        x, _ = load_eval_pairs(ddir, args.n_scale_pairs, 1, seed=meta.seed)
        states = x[:, -1]
        background = float(np.mean(states))
        scale = float(np.median(np.std(states, axis=1)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"non-positive data scale for {condition} seed {meta.seed}")
        n = states.shape[-1]
        valid_modes = [k for k in modes if 0 < k <= n // 2]
        grid = np.arange(n)
        samples, meta_rows = [], []
        for amp_factor in amplitudes:
            amp = amp_factor * scale
            for k in valid_modes:
                for phase in phases:
                    samples.append(background + amp * np.cos(2 * np.pi * k * grid / n + phase))
                    meta_rows.append((amp_factor, amp, k, phase))
        arr = np.asarray(samples, dtype=np.float32)[:, None, :]
        base = np.full((1, 1, n), background, dtype=np.float32)
        model = load_model(path, args.device, args.model_factory)
        with torch.no_grad():
            y0 = model_output(model, to_batch(base, args.device)).cpu().numpy()[0]
            y = model_output(model, to_batch(arr, args.device)).cpu().numpy()
        response = y - y0[None]
        input_delta = arr[:, 0] - background
        xf = np.fft.fft(input_delta, axis=1)
        yf = np.fft.fft(response, axis=1)
        ref = reference_symbol(pred, condition, n)
        grouped = {}
        for i, (amp_factor, amp, k, phase) in enumerate(meta_rows):
            ratio = yf[i, k] / xf[i, k]
            total = float(np.sum(np.abs(yf[i, 1:]) ** 2))
            fundamental = float(np.abs(yf[i, k]) ** 2)
            if k != n - k:
                fundamental += float(np.abs(yf[i, n - k]) ** 2)
            harmonic = max(total - fundamental, 0.0) / total if total > 0 else np.nan
            grouped.setdefault((amp_factor, amp, k), []).append((ratio, harmonic))
        for (amp_factor, amp, k), values in grouped.items():
            ratios = np.asarray([v[0] for v in values])
            harmonics = np.asarray([v[1] for v in values], dtype=float)
            z = ratios.mean()
            target = ref[k]
            records.append({
                "pde": meta.pde, "r": meta.r, "condition": condition,
                "seed": meta.seed, "reference_scope": reference_scope(meta.pde),
                "mode": k, "normalized_frequency": k / (n // 2),
                "amplitude_factor": amp_factor, "amplitude": amp,
                "response_real": z.real, "response_imag": z.imag,
                "response_magnitude": abs(z), "response_phase": np.angle(z),
                "reference_real": target.real, "reference_imag": target.imag,
                "reference_distance": abs(z - target), "identity_distance": abs(z - 1),
                "phase_equivariance_cv": float(np.std(ratios) / (abs(z) + 1e-12)),
                "harmonic_power_fraction": float(np.nanmean(harmonics)),
                "data_scale": scale,
            })
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[{index}/{len(paths)}] probed {condition} seed {meta.seed}")
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--jacobian-profiles", required=True)
    ap.add_argument("--predictors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default=None,
                    help="optional checkpoint glob enabling finite-amplitude probes")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--data-template", default="{pde}_r{r}")
    ap.add_argument("--condition-regex", default=CONDITION_REGEX)
    ap.add_argument("--run-regex", default=C.DEFAULT_RUN_REGEX)
    ap.add_argument("--model-factory", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--probe-modes", default="1,2,4,8,16,24,31")
    ap.add_argument("--probe-amplitudes", default="0.25,1.0")
    ap.add_argument("--n-phases", type=int, default=4)
    ap.add_argument("--n-scale-pairs", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    metrics, points = analyse_jacobians(args.jacobian_profiles, args.predictors)
    summary = summarise(metrics)
    metrics.to_csv(os.path.join(args.out, "fourier_metrics.csv"), index=False)
    points.to_csv(os.path.join(args.out, "fourier_symbols.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "fourier_summary.csv"), index=False)
    plot_symbols(points, os.path.join(args.out, "fourier_symbols.png"))

    show = ["pde", "r", "reference_scope", "n_seeds",
            "rmse_reference_mean", "rmse_identity_mean",
            "reference_advantage_mean", "fraction_modes_closer_reference_mean",
            "rmse_reference_low_mean", "rmse_reference_high_mean"]
    print(summary[show].round(4).to_string(index=False))
    print("\nreference_advantage > 0 means closer to the saved PDE reference than identity.")
    print("Wave uses only the saved spatial part of the leapfrog stencil, not the full propagator.")

    if args.runs:
        probes = probe_checkpoints(args, np.load(args.predictors))
        probes.to_csv(os.path.join(args.out, "sinusoid_probes.csv"), index=False)
        print(f"wrote {len(probes)} finite-amplitude probe rows")
    print(f"\nwrote Fourier outputs to {args.out}")


if __name__ == "__main__":
    main()
