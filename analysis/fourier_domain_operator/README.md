# Fourier-domain operator audit

This analysis compares the effective operator recovered by the Jacobian
analysis with the saved analytical PDE stencil in frequency space. It creates
new derived outputs only; it does not modify training, checkpoints, upstream
analysis outputs, or require retraining.

The primary evidence comes from `jacobian_profiles.npz`. For each checkpoint,
the signed-offset Jacobian profile is embedded on the periodic ring and Fourier
transformed. Errors against the saved PDE reference and against identity are
computed over non-DC modes, with low/middle/high-frequency breakdowns.

```cmd
python analysis\fourier_domain_operator\selftest.py

python analysis\fourier_domain_operator\fourier_analysis.py ^
  --jacobian-profiles analysis\is_it_mechanistic\results\jacobian_profiles.npz ^
  --predictors analysis\what_do_heads_learn\predictors.npz ^
  --out analysis\fourier_domain_operator\results
```

To add finite-amplitude sinusoidal probes of every checkpoint:

```cmd
python analysis\fourier_domain_operator\fourier_analysis.py ^
  --jacobian-profiles analysis\is_it_mechanistic\results\jacobian_profiles.npz ^
  --predictors analysis\what_do_heads_learn\predictors.npz ^
  --out analysis\fourier_domain_operator\results ^
  --runs "final_checkpoints/**/best_model.pt" ^
  --data-root data ^
  --model-factory model_adapter:build_model_from_checkpoint ^
  --device cuda:0
```

## Interpretation boundaries

- `reference_advantage > 0` means the learned symbol is closer to the saved
  stencil reference than to identity; a negative value means the reverse.
- The wave `stencil.npy` contains only `[r, 2-2r, r]`, the spatial part of a
  leapfrog update. Wave comparisons are marked `spatial_part_only` and are not
  evidence about recovery of the complete two-time-level propagator.
- Sinusoids are controlled probes, not necessarily in-distribution samples.
  Their amplitude is scaled to held-out data and their results remain separate
  from the local Jacobian analysis.
- Fourier-transforming attention profiles would be descriptive only and is not
  used here as a substitute for the model's input-output transfer function.

## Outputs

- `fourier_metrics.csv`: one row per checkpoint.
- `fourier_symbols.csv`: frequency-resolved learned/reference symbols.
- `fourier_summary.csv`: condition-level means and seed intervals.
- `fourier_symbols.png`: nine-panel magnitude comparison.
- `sinusoid_probes.csv`: optional finite-amplitude responses.
