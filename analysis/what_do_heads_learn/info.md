# What do heads learn?

Gated three-way model selection over attention heads: stencil vs.
autocorrelation vs. band-limited spectral computation.

## Run order

```bash
python selftest.py                         # verify before touching real data
python selftest.py --sweep                 # noise tolerance

python extract_attention_profiles.py --runs "final_checkpoints/**/best_model.pt" --dry-run
python extract_attention_profiles.py --runs "final_checkpoints/**/best_model.pt" \
    --data-root data --out profiles/trained \
    --model-factory model_adapter:build_model_from_checkpoint
python extract_attention_profiles.py --runs "baselines/**/init.pt" \
    --data-root data --out profiles/baseline --baseline \
    --model-factory model_adapter:build_model_from_checkpoint

python build_predictors.py --data-root data --out predictors.npz \
    --stencil-offsets="-1,0,1"

python assign_heads.py --trained profiles/trained --baseline profiles/baseline \
    --predictors predictors.npz --out results --half-width 4
```

Note `--stencil-offsets="-1,0,1"` needs the `=`; argparse reads a bare
`-1,0,1` as a flag. All parsers use `allow_abbrev=False`, so full flag names
are required and the `--r` / `--resume_from` collision cannot recur.

## Three seams, all resolved for this project's VanillaTransformer

Two are in `extract_attention_profiles.py` and marked `ADAPTER`; the third is the
attention-capture routine in `attention_common.py`. All three are wired up via
`model_adapter.py` in this directory and verified end-to-end against a real
checkpoint (`final_checkpoints/heat_new_0.25/seed_100/best_model.pt`) --
correct architecture reconstruction, correct attention capture (rows sum to
1 exactly), correct held-out data resolved for that exact seed.

1. `load_model` turns a checkpoint into a module. This project's checkpoints
   (`best_model.pt` / `final_model.pt` / `extracted_model.pt`) are all
   state_dicts, not pickled modules, so `--model-factory
   model_adapter:build_model_from_checkpoint` is required. It rebuilds
   `VanillaTransformer` from `ckpt["args"]` and infers `NX` from the
   `embedding.pe` buffer shape rather than hardcoding 64. `transformer.py`
   itself cannot be used as the factory target: it runs
   `argparse.parse_args()` and starts loading data at module scope with no
   `__main__` guard, so importing it executes the training script.
2. `load_inputs` / `to_batch` supply held-out snapshots. Default assumes
   `(B, N)`, which is exactly what `VanillaTransformer.forward` expects, so
   no change is needed here. `condition_data_dir` (which locates the data
   directory `load_inputs` reads from) needed real work, covered below.
3. Attention capture patches `nn.MultiheadAttention.forward` and
   `F.scaled_dot_product_attention`. This did **not** work for this
   project's model when first tried: `nn.TransformerEncoderLayer` takes an
   internal fast path in eval mode that calls neither of those from Python,
   so the patches captured nothing and `extract_attention()` raised "no
   attention captured". Fixed at the source: `attention_common.force_slow_path_patches`
   (carried over from the mechanistic analysis, which hit the same problem) replaces
   `TransformerEncoderLayer.forward`/`TransformerEncoder.forward` with
   explicit equivalents that route through `MultiheadAttention`, and
   `AttentionCapture.__enter__` now applies it automatically. The default
   `extract_attention` (no `--attention-extractor` flag needed) is verified
   working directly against a real checkpoint as of this fix.
   `model_adapter.py`'s hook-based `extract_attention` (the original,
   independent workaround) still exists and still works, but is redundant
   now; pass `--attention-extractor model_adapter:extract_attention`
   only if you want the second, independently-implemented capture path as a
   cross-check.

## Real directory-naming issues found by running this against actual data

These were invisible from reading the code; each one only showed up by
actually running the pipeline against `final_checkpoints/` and `data/` as
they exist on disk right now, mid-training.

**Checkpoint and data directory names don't match a fixed template.** Heat's
convention is `heat_new_<r>` (and an older, still-present `heat_<r>`); LF and
wave's is `<scheme>_r_<r>` with a doubled separator around `r`. Neither
matched the original `DEFAULT_RUN_REGEX`/`CONDITION_REGEX` at all -- a dry
run against the real checkpoint tree parsed 0 of 60 checkpoints before this
was fixed. The regex now handles both conventions and was verified against
every directory actually present under `data/` and `final_checkpoints/`,
including confirming that the unrelated `heat_nlayers_*` sweep still does
not match.

**Real data lives per-seed, not per-condition.** `run_heat.py` and
`run_friedrichs.py` both point the generator's `--save_dir` at
`<condition>/seed_<N>`, so e.g. `data/heat_new_0.25` itself has no files,
only `seed_100/`, `seed_101/`, ... subdirectories, each with its own
`stencil.npy`/`train.npy`/`test.npy` (stencil.npy is identical across seeds,
since it depends only on r, but is saved once per generation run anyway).
`build_predictors.py` and `../parameter_sensitive/cross_correlation.py` build population-level predictors
and are free to use any one seed's data (`resolve_condition_dir`, preferring
the flat layout if present, else the first seed found).
`extract_attention_profiles.py` must match a specific checkpoint to *its own*
seed's held-out data, not an arbitrary one (`condition_data_dir`).

**More than one directory can parse to the same (pde, r), and they are not
guaranteed to agree.** Both `heat_0.25` and `heat_new_0.25` exist on disk.
Confirmed by inspection: `heat_0.25/stencil.npy` (and every seed under it)
actually holds an `r=0.1024` stencil despite the directory's name -- this is
a real problem with that data, not something either script can safely paper
over. `build_predictors.py`/`../parameter_sensitive/cross_correlation.py` now warn on stderr when a
`(pde, r)` key is about to be silently overwritten by a second directory,
rather than picking one with no trace of the collision.
`extract_attention_profiles.py`'s `condition_data_dir` avoids the ambiguity
entirely for the checkpoint-matching case, since it can reuse the
checkpoint's own directory name.

**Data generation finishes per-condition, not all at once.** As of this
writing, `heat_new_0.4`, `lf_r0.31`, `lf_r0.4`, and all four `wave_r_*`
directories exist but are empty. `build_predictors.py` and
`../parameter_sensitive/cross_correlation.py`
skip an empty condition with a warning and keep going, rather than crashing
the whole run -- expected, not exceptional, given training completes
incrementally.

## Remaining gap: no baseline checkpoints exist yet

Not an analysis code issue: nothing in the repo currently writes `init.pt`-style
baseline checkpoints to disk (`compute_random_baselines.py` builds untrained
models in-memory and never saves them). The `--baseline`/`profiles/baseline`
step in the run order above needs baseline checkpoints to glob over before
it can run; either save N untrained `VanillaTransformer` state_dicts to a
`baselines/` tree in the same `{"model_state_dict": ..., "args": ...}`
format, or point `--runs` at whatever convention you choose.

## Design decisions the self-test forced

Each of these was a bug that the synthetic recovery test caught, and each
would have been invisible on real data.

**Selection and validity use different scales.** Selection compares one head
against three targets, so raw divergence is the right scale. Validity compares
heads against a fixed target, where the nulls differ sharply, so it uses
`z = (d - median_base) / sigma_base` per predictor. Mixing these breaks in both
directions: a pooled floor lets the most permissive predictor set the
threshold and sends genuine stencil heads to `no_fit`, while z-scored
selection over-rewards whichever hypothesis has the widest null.

**The spectral grid must be pruned from below.** A band-limited operator row
has wavelength ~N/kmax, so at small kmax it is flat across a narrow window and
collapses onto the uniform distribution. Unpruned, uniform random-init heads
fit "spectral" at JS ~3e-4 and the label becomes a synonym for "unstructured".
`--min-structure` drops those kmax values, which ties usable bandwidth to
window width rather than leaving it a free choice. If the grid empties, widen
`--half-width`.

**ACF and spectral may not be separable.** If initial conditions are generated
as band-limited random fields, the empirical ACF *is* approximately a
band-limited profile. `predictor_separation` measures this per condition, and
where `d_acf_spectral` falls below `--min-structure` the two labels merge into
`acf_or_spectral` automatically. Expect separation to survive mainly in the
Lax-Friedrichs arm: a spatial ACF of a stationary field is necessarily even,
while the LF operator row is asymmetric, so LF is the discriminator for this analysis in
the same way it is for the paper as a whole. Heat and wave will likely report
the merged label, which is the honest outcome and leaves the primary contrast
(stencil vs. not) intact.

**Failure mode is conservative.** The jitter sweep shows accuracy degrading
into `no_fit` rather than into confident wrong labels: 100% at 2% profile
noise, 78% at 10%, and effectively all `no_fit` by 25%. Under-labelling is the
expected failure, so a low labelled fraction on real data means the floor is
strict, not that heads are unstructured. Report `--floor-pct` sensitivity.

**In-window mass is an alignment check.** `assign_heads.py` warns when median
in-window mass drops below 0.2, which is the signature of `profile[..., i]` not
corresponding to `offsets[i]`. During development this exact misalignment made
concentrated heads look uniform and matched the baseline divergences to four
decimals.

## Outputs

- `heads.csv` — one row per (pde, r, seed, layer, head) with raw divergences,
  calibrated z-scores, fitted kmax, entropy, in-window mass, and label. This is
  the join key for historical head-ablation analyses, where ablation cost is
  cross-referenced against the attention label.
- `label_fractions.csv`, `head_labels.png/pdf` — the headline stacked bar.
- `predictor_separation.csv` — identifiability per condition; report this in
  the appendix, since it bounds what the three-way split can claim.
- `baseline_heads.csv` — the null, for the floor sensitivity analysis.
- `spectral_kmax_distribution.csv`, `spectral_kmax_distribution.png/pdf` — counts
  of fitted kmax per (pde, r) among heads labelled `spectral`, the
  diagnostic that distinguishes a genuine low-pass head (fit at the low end
  of the grid) from one nesting toward the stencil (fit at the top). Empty
  if every head labelled `spectral` merged into `acf_or_spectral` instead.

## Wave caveat

This analysis compares spatial attention within a timestep uniformly across all three
PDEs, so wave's `stencil.npy` holding only `[r, 2-2r, r]` is consistent with
that framing. It still needs the footnote noting the absent `u^{n-1}`
coefficient.

No wave-equation data generator exists in this repo yet (only
`generate_heat_equation.py` and `generate_lax_friedrichs.py` do); the wave
arm of the headline figure can't be produced until one is added and run.

## References

- Kullback, S., & Leibler, R. A. (1951). On information and sufficiency.
  *Annals of Mathematical Statistics*, 22(1), 79-86. — `kl_divergence`.
- Lin, J. (1991). Divergence measures based on the Shannon entropy. *IEEE
  Transactions on Information Theory*, 37(1), 145-151. — `js_divergence`,
  the default `--divergence` for both selection and the floor.
- Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the median absolute
  deviation. *Journal of the American Statistical Association*, 88(424),
  1273-1283. — the 1.4826 MAD-to-sigma constant in `assign._robust_scale`.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
  rate: a practical and powerful approach to multiple testing. *Journal of
  the Royal Statistical Society: Series B*, 57(1), 289-300. — the BH
  correction behind the "2/84 problem" this pipeline replaces (see
  `assign_heads.py` module docstring).
- LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and
  Partial Differential Equations*. SIAM. — von Neumann/Fourier-symbol
  analysis of a finite-difference operator, the basis for `spectral_row`'s
  truncated-symbol construction; already the standing reference for the
  stencils themselves (see `generate_heat_equation.py`,
  `generate_lax_friedrichs.py`).
