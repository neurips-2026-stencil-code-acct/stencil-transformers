# Stencil Transformers paper reproducibility snapshot

This folder is the readable, paper-scoped subset of the parent repository. It
contains the code, compact intermediate arrays, saved tables, validation tests,
and final figures needed to inspect the heat-equation and Lax--Friedrichs
experiments. The original project files are unchanged.

AI usage in generated code, verified by the authors.

The large generated datasets and trained checkpoints are intentionally not
duplicated. Together they occupy about 93 GB in the parent project. Their
expected locations and directory layout are documented in
[`EXTERNAL_INPUTS.md`](text_files/EXTERNAL_INPUTS.md).

## Fast verification

From PowerShell in this folder:

```powershell
& .\run_files\run_checks.ps1
```

This runs the synthetic tests for attention extraction, parameter tracking,
attention replacement, Jacobian/Fourier conventions, robustness calculations,
and the constrained-model positive control.

To recreate the two paper figures from the included tables:

```powershell
& .\run_files\make_figures.ps1
```

To recompute only the inexpensive downstream summaries from the included
intervention, Jacobian, and Fourier tables:

```powershell
& .\run_files\rebuild_derived_results.ps1
```

Neither command trains a model or changes files outside this folder.

## Workflow represented here

1. `generate_data/generate_heat_equation.py` and
   `generate_data/generate_lax_friedrichs.py` create the two datasets.
2. `transformer.py` trains the ordinary transformer. `analysis/metrics.py`
   computes task-performance diagnostics.
3. `analysis/what_do_heads_learn` extracts attention patterns and compares them with calibrated
   reference patterns.
4. `analysis/parameter_sensitive` tests whether attention summaries change with the equation
   parameter.
5. `analysis/is_it_mechanistic` replaces attention patterns and extracts the full-model
   Jacobian.
6. `analysis/fourier_domain_operator` compares the Jacobian's Fourier response with the known PDE
   update and with copying the input.
7. `analysis/robustness_checks` resamples model seeds and audits the update
   Jacobian after subtracting identity.
8. `analysis/positive_control` repeats the core diagnostics in a constrained
   model where attention is the only spatial mixing mechanism.
9. `analysis/workshop_figures` and `analysis/positive_control/plot.py` render
   the two current paper figures.

[`CODE_MAP.md`](text_files/CODE_MAP.md) gives a file-by-file guide and marks legacy
compatibility code that is retained only because a downstream analysis reads
its saved output.

## Paper scope

The current paper reports only the heat and Lax--Friedrichs conditions, with
three parameter values and 20 trained initializations per condition. Some
source result files also contain historical wave-equation rows because they
are exact snapshots of shared upstream tables. Every paper-facing export,
robustness calculation, figure, and validation check filters to `heat` and
`lf`.

The older clustering, specialization, wave, and head/layer-ablation branches
are not copied here because they are not part of the current argument.

## Terminology in machine-readable files

Several CSV column values retain old internal names so the analysis scripts
remain compatible:

- `mse_persistence` is the mean squared error obtained by copying the input as
  the next-step prediction.
- `analytical_stencil` is an internal replacement label for the known
  three-point finite-difference update weights.
- `numerical_stencil` may appear in older source comments and means the same
  finite-difference weights stored by the data generator.

These are implementation identifiers, not recommended manuscript wording.
The README files use “copy-the-input baseline,” “known update weights,” and
“attention replacement” where possible.

## Refreshing this snapshot

`run_files/refresh_from_parent.ps1` recopies the explicitly listed shared
implementation files, portable inputs, and small outputs from the parent
repository. The renamed analysis modules in this package remain authoritative
because copying the parent's legacy module names would undo the refactor. The
script maps legacy result paths into the descriptive layout, does not delete
anything, and rebuilds the hash manifest.

```powershell
& .\run_files\refresh_from_parent.ps1
& .\run_files\run_checks.ps1
```

If you intentionally edit a packaged file, run `build_manifest.py` after the
edit so the integrity manifest describes the new snapshot.
