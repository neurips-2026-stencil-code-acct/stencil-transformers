# Code and evidence map

The package uses descriptive analysis-directory and module names. The refresh
script maps the parent repository's legacy paths into this layout.

| Stage | Primary files | Included evidence | Role in the paper |
|---|---|---|---|
| Data | `generate_data/generate_heat_equation.py`, `generate_data/generate_lax_friedrichs.py` | Generated arrays remain external | Produce periodic one-dimensional trajectories and store the known update weights |
| Training | `transformer.py`, `analysis/metrics.py`, `check_convergence.py` | Checkpoints remain external | Train and evaluate the ordinary transformer |
| Random controls | `compute_random_baselines.py` | `analysis/what_do_heads_learn/profiles/baseline`, random-control comparison tables | Calibrate attention statistics against untrained models |
| Attention patterns | `analysis/what_do_heads_learn/extract_attention_profiles.py`, `build_predictors.py`, `assign_heads.py` | `profiles`, `predictors.npz`, `results` | Extract head-wise attention patterns and apply the prespecified gates |
| Parameter tracking | `analysis/parameter_sensitive/cross_correlation.py`, `attention_statistics.py`, `parameter_regression.py` | `xcorr.npz`, `results` | Compare measured attention summaries with equation- and data-derived trends |
| Full-model Jacobian | `analysis/is_it_mechanistic/jacobian_analysis.py`, `mechanistic_model.py` | `results/jacobian.csv`, `jacobian_profiles.npz` | Recover the local input-output operator and its linearity/circulance diagnostics |
| Attention replacement | `analysis/is_it_mechanistic/head_preserving_substitution.py`, `model_review_fix.py` | `results/head_preserving` | Compare intact attention with validation-derived fixed patterns and known update weights while preserving head identity |
| Fourier audit | `analysis/fourier_domain_operator/fourier_direction.py`, `support_aware_analysis.py`, summary scripts | `results` | Evaluate the learned operator on supported, near-support, and extrapolation frequencies |
| Uncertainty and controls | `analysis/robustness_checks` | Seed-level contrasts and 20,000-resample summaries | Preserve within-model pairing and use model initialization as the resampling unit |
| Detectability control | `analysis/positive_control` | 120 model records, summaries, and Figure 1 | Check whether the diagnostics recover known update weights when attention is the only spatial mixing route |
| Figures | `analysis/positive_control/plot.py`, `analysis/workshop_figures/figure4_operator_robust.py` | `figures/Figure1_detectability.*`, `figures/Figure2_fourier_response.*` | Recreate the current paper figures from saved tables |

## Important compatibility files

- `analysis/is_it_mechanistic/attention_substitution.py` and
  `analysis/is_it_mechanistic/results/substitution.csv` support the historical all-layer
  comparison read by `bootstrap_model_effects.py`. They are not the paper's
  primary head-preserving replacement analysis.
- `analysis/fourier_domain_operator/fourier_analysis.py` is retained because
  the corrected direction wrapper imports it. Use `fourier_direction.py` as the public entry
  point.
- `reference_launchers` contains snapshots of the old GPU launchers only to
  preserve training provenance. They contain machine-specific GPU scheduling
  and cleanup logic and should not be run directly.

## Deliberate exclusions

- Raw `data/` arrays and `final_checkpoints/` are external because of size.
- Clustering and specialization code is absent because the current paper does
  not make a cross-seed head-identity claim.
- Head and layer ablations are absent because they are not used in the
  current paper.
- Older workshop Figure 3 scripts and manuscript-integration scratch files
  are absent because the current draft has a two-figure evidence chain.
- Smoke-test outputs are absent. Synthetic test code is included, but only the
  full saved outputs are packaged as evidence.
