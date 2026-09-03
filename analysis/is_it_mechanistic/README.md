# Is it mechanistic?

Builds on the attention and parameter-sensitive analyses
(`analysis/what_do_heads_learn/info.md` and
`analysis/parameter_sensitive/README.md`).
`mechanistic_model.py`, `jacobian_analysis.py`, `attention_substitution.py`,
and `selftest.py` all import `attention_common` (and the
jacobian/substitution scripts also import
`build_predictors`/`extract_attention_profiles`)
directly from `../what_do_heads_learn`. Each file adds that path to `sys.path` itself, so
there is a single copy of that code, not a duplicate kept in sync by hand —
a duplicate `attention_common.py` used to live in this directory and was already
missing fixes from the shared copy by the time it was found.

The upstream analyses read saved attention patterns. This analysis runs the model, so it needs the model-loading
seams and, unlike everything before it, it has to reach inside the attention
computation.

## Run order

```bash
python selftest.py                          # verify before real data

python jacobian_analysis.py --runs "../../final_checkpoints/**/best_model.pt" \
    --data-root ../../data --predictors ../what_do_heads_learn/predictors.npz --out results \
    --model-factory model_adapter:build_model_from_checkpoint
python attention_substitution.py --runs "../../final_checkpoints/**/best_model.pt" \
    --data-root ../../data --predictors ../what_do_heads_learn/predictors.npz --xcorr ../parameter_sensitive/xcorr.npz \
    --out results --model-factory model_adapter:build_model_from_checkpoint
```

Paths above assume the upstream analyses' `--out` locations from their run orders;
adjust to wherever those were actually written. `--model-factory` is
required for the same reason it is in the attention analysis: this project's checkpoints are
state_dicts, not pickled modules (see `analysis/what_do_heads_learn/info.md`).

All four need `(n_traj, n_t, N)` data so inputs pair with the next step. None
needs retraining.

## The fast-path problem, which would have silently voided everything

In eval mode `nn.TransformerEncoderLayer` dispatches to a fused kernel that
computes attention internally, bypassing both `MultiheadAttention.forward` and
`scaled_dot_product_attention`. Nothing is interceptable, and the symptom is
not an error: capture returns empty and every intervention becomes a no-op
that reports as a clean null result. The smoke test hit this on the first real
model.

`force_slow_path_patches` in attention_common replaces
`TransformerEncoderLayer.forward` and `TransformerEncoder.forward` with their
explicit equivalents, which route through MultiheadAttention. The self-test
asserts the replacement is numerically identical to PyTorch's own output on a
real `nn.TransformerEncoder`: it comes out at exactly 0.0 absolute error.
Re-verified after merging into this project's
`analysis/what_do_heads_learn/attention_common.py`: the `selftest.py`
fidelity check still reports exactly `0.000e+00` against a real
`nn.TransformerEncoder`, and a live run of `jacobian_analysis.py` against a
real checkpoint (`final_checkpoints/heat_new_0.25/seed_100`) now captures
attention where it previously raised "no attention captured".

This also affects attention extraction: `AttentionCapture` had the same blind spot and now
carries the same fix, merged directly into
`analysis/what_do_heads_learn/attention_common.py` (not
kept as a separate mechanistic-analysis copy). If attention extraction ran before this
change and raised "no attention captured", that was why;
`model_adapter.py`'s hook-based `extract_attention` was an independent
workaround for the same bug and still works, but is redundant now that the
fix is upstream.

Every intervention calls `assert_intercepted`. A substitution that never
reached the model and a substitution that genuinely does not change the loss
produce identical numbers, and only one of them is a result.

## A second real issue found only by running this against actual data

`condition_data_dir` (imported from `extract_attention_profiles.py`, same as
the head-interpretation analysis uses)
resolves each checkpoint to held-out data matching its own seed — necessary
here for the same reason it matters in the attention analysis, and more visibly so: this
project's training scripts delete a seed's generated data after that seed
finishes training, to save disk (see `compute_random_baselines.py`'s
`--regenerate_data` option and its docstring). Checkpoint seeds and remaining
data seeds routinely don't overlap at all: confirmed on real data, where
`lax_friedrichs_r_0.1` has checkpoints for seeds 300-319 but data only for
seed 300, and `lax_friedrichs_r_0.25`'s checkpoints (300-309) and data
(305-309) barely overlap. `condition_data_dir` now falls back to any other
available seed of the same (pde, r) rather than failing outright -- still
statistically valid, since every seed of a condition is an independent draw
from the same generator, but logged as a warning rather than applied
silently, since it means evaluating a checkpoint against a held-out split it
did not itself produce.

## The Jacobian

Three quantities, all needed for the claim to hold:

- `r2_linearity` — how well the local linearisation predicts held-out outputs.
  A Jacobian is the operator only if the map is linear, and softmax, LayerNorm
  and the MLP are not. Below 0.95 the script warns, because "the model
  recovers the stencil" does not follow from a matching row when the map is
  not linear.
- `circulant_residual` — the true operator is translation invariant. A
  non-circulant Jacobian is not a finite-difference operator whatever its
  rows look like, and the diagonal-averaged row is then a summary rather than
  the thing itself.
- `stencil_rel_error` — absolute, not correlation. Correlation is shape-only
  and would score a scaled or near-uniform operator as perfect, which is the
  same trap flagged in the attention-analysis methodology notes.

Multi-step inputs are split per timestep, so a leapfrog model receiving
`(u^{n-1}, u^n)` yields both blocks and the `u^{n-1}` block can be checked
against its predicted coefficient of -1.

The self-test recovers `[0.25, 0.5, 0.25]` and `[0.625, 0, 0.375]` to zero
error with zero circulant residual, so a null result on real checkpoints is
about the model, not the extraction.

**A live run on two real checkpoints found exactly that kind of null, worth
flagging before anyone treats "the Jacobian recovers the stencil" as settled.**
`heat_new_0.25/seed_100` and `lax_friedrichs_r_0.25/seed_300` both linearise
almost perfectly (`r2_linearity` = 0.9998 and 1.0, `circulant_residual` =
0.0008 and 0.0053 — the extraction is trustworthy on both) but the recovered
operator in both cases sits close to the identity (`coef_+0` ~ 0.98-0.99,
flanking coefficients near 0, `row_sum` 0.86-0.96) rather than either PDE's
true stencil (`stencil_rel_error` 0.98 and 1.68; `stencil_corr` 0.81 and
-0.02). That is two checkpoints on a quick smoke setting (4 base points, 32
linearity probes) — nowhere near enough to write up as the paper's finding —
but it is a consistent, cross-PDE signal that these particular models may be
closer to a persistence-like predictor than to the diffusion/advection
operator, and it is corroborated by 4b on the same checkpoint: freezing
attention only costs 33% of the original-to-persistence gap (moderate, not
near-zero), while substituting the *true* stencil is catastrophic (excess
error ~5, ~3000x the original MSE) -- consistent with a model whose value/
output projections were tuned around something other than the diffusive
stencil pattern. Worth checking convergence (val loss) on these checkpoints
and rerunning the Jacobian analysis with full settings across more seeds before drawing a
conclusion either way.

## Attention substitution

The `frozen` control decides how to read every other row. It substitutes the
model's own attention averaged over the test set, i.e. its own pattern stripped
of input dependence. If freezing costs nothing, attention is effectively a
fixed matrix, no routing is happening, and the comparison is about which fixed
matrix works. If freezing is costly, no fixed substitute can match the original
and the other substitutions must be read against `frozen`, not against
`original`. Without this control, "every substitution degrades accuracy" is
ambiguous between "attention carries the operator" and "attention carries
input-dependent information of some other kind".

On the smoke models the pattern came out as the thesis predicts: acf and xcorr
substitution cost essentially the same as freezing (0.267 against 0.258) while
stencil substitution costs four times more. That is the direct causal form of
the claim, and it is worth noting that it is visible on models this small.

## Excluded head and layer ablations

Head ablation is mean ablation, not zeroing. Zeroing removes the head's
constant contribution too, conflating "carries no information" with "carries a
constant the rest of the network depends on" -- the same distinction drawn
between zero-ablation and resample/mean-ablation in the mechanistic
interpretability literature (Wang et al., 2023; see References).

Costs do not decompose additively: heads within a layer are summed through a
shared output projection and can compensate for one another, so no total is
reported.

Historical ablation code joined by attention label rather than by head index.
That code is not included in this paper-scoped package because the current
paper does not report the head- or layer-ablation branch.

## Reading `excess`

`excess` normalises by the model's skill margin over persistence, so 0 is
intact and 1 is as bad as predicting no change -- the mean-squared-error skill
score of forecast verification (Murphy, 1988; see References), here scoring
the model against the persistence forecast rather than against climatology.
Values above 1 are worse than persistence and are unbounded: a full layer
ablation on a 2-layer model reached 779x the original MSE on the smoke run,
which is a real effect rather than a metric failure.

The normaliser can be unstable, though. All three schemes approach the identity
at small r, so persistence is a strong baseline and the margin can become a
thin difference of similar numbers. `skill_guard` warns below a 10% margin and
directs the reader to the `ratio` column, which stays interpretable. Worth
watching on the heat r=0.1 arm in particular.

## Outputs

- `jacobian.csv`, `jacobian_profiles.npz` — recovered operator per run and
  input timestep, with linearity and circulance diagnostics.
- `substitution.csv` — MSE, excess and ratio for every substitution, all
  layers and optionally per layer.

## The dissociation table

The paper's strongest single artifact pairs the Jacobian and attention analyses: the
Jacobian's distance from the stencil next to the attention profile's distance
from the stencil, per condition. Both numbers already exist, in
`jacobian.csv` and `../what_do_heads_learn/results/heads.csv`. If the first is small and the second
is large, the result is not that attention fails to look like the operator but
that the operator is recovered elsewhere, and the paper's contribution changes
from a negative claim about attention to a positive one about where the
computation resides.

## References

- Kriegeskorte, N., Simmons, W. K., Bellgowan, P. S. F., & Baker, C. I.
  (2009). Circular analysis in systems neuroscience: the dangers of double
  dipping. *Nature Neuroscience*, 12(5), 535-540. — the general reason
  `assert_intercepted` exists: a null result and an instrumentation failure
  must not be allowed to look identical.
- Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J.
  (2023). Interpretability in the wild: a circuit for indirect object
  identification in GPT-2 small. *ICLR 2023*. — mean/resample ablation over
  zero-ablation, for the same reason given in `head_output_means`/
  `AttentionControl`'s docstrings.
- Murphy, A. H. (1988). Skill scores based on the mean square error and
  their relationships to the correlation coefficient. *Monthly Weather
  Review*, 116(12), 2417-2424. — the skill-score form of `excess`/
  `skill_guard`, here against a persistence rather than climatological
  reference forecast.
