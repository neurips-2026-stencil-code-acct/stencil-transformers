# Is the learned structure parameter-sensitive?

Builds on the attention-pattern analysis
(`analysis/what_do_heads_learn/info.md`) and covers the parameter-sensitive
tests. `experimental_design.py`, `attention_statistics.py`, and
`cross_correlation.py` import `attention_common`, `assign_heads`, and
`build_predictors` directly from
`../what_do_heads_learn` (each adds that directory to `sys.path` itself), so there is a
single copy of that code, not a duplicate kept in sync by hand.

## Run order

```bash
python selftest.py                          # verify before real data

python experimental_design.py --validate --predictors ../what_do_heads_learn/predictors.npz \
    --ic-spectrum-from ../what_do_heads_learn/data/heat_r0.25      # run BEFORE spending compute
python experimental_design.py --pde lf --have 0.1,0.25,0.4 \
    --candidates 0.5,0.6,0.7,0.8,0.9 --add 2

python cross_correlation.py --data-root ../what_do_heads_learn/data --out xcorr.npz
python attention_statistics.py --profiles ../what_do_heads_learn/profiles/trained --predictors ../what_do_heads_learn/predictors.npz \
    --xcorr xcorr.npz --head-labels ../what_do_heads_learn/results/heads.csv --out results/stats.csv
python parameter_regression.py --stats results/stats.csv --out results
```

Paths above assume the attention analysis's `--out`/`--data-root` locations from its run
order (`analysis/what_do_heads_learn/info.md`); adjust to wherever those were actually
written if you pointed that analysis elsewhere.

`cross_correlation.py` needs the time axis intact, i.e. `(n_traj, n_t, N)`. Consecutive
pairing is the entire point, so a flattened snapshot array will not do.

## Statistics

Four bounded statistics, computed identically on attention profiles and on
predictor profiles, so a fitted slope of exactly 1 with zero intercept means
quantitative tracking rather than mere monotonicity:

    centrality  p[0] / (p[-1] + p[0] + p[+1])
    asymmetry   (p[+1] - p[-1]) / (p[-1] + p[+1])
    width       sqrt(sum_d p[d] d^2)
    shift       sum_d p[d] d

Bounded forms replace log ratios because the LF stencil has an exact zero
centre, which sends any log-ratio statistic to -inf on the arm that matters
most. Under these definitions the stencil predictions come out linear:
heat centrality = 1 - 2r, wave centrality = 1 - r, LF asymmetry = -nu. The
smoke test reproduces all three to four decimals.

## Design decisions the self-test forced

**The ACF cannot address asymmetry at all.** A circular spatial ACF satisfies
rho[d] = rho[-d] exactly, so on the LF arm the ACF hypothesis predicts zero
asymmetry at every nu no matter what the data contain. Left as-is, the
headline arm could only reject operator tracking, never identify what replaced
it. `cross_correlation.py` adds the input-output cross-correlation, which is asymmetric
whenever the scheme advects and is the correct data-side competitor. Its heat
and wave asymmetries come out at exactly 0, which doubles as a correctness
check on the sign conventions.

**The joint model is saturated at three parameter values.** With K parameter
values and P predictors the between-condition fit has K - P - 1 residual
degrees of freedom; at K = 3, P = 2 that is zero, and the self-test confirms
r2_between = 1.000000 under *both* generators. The joint partial coefficients
are reported but are not evidence. The single-predictor comparison is the test.

**Correlation is the wrong identifiability metric, and this nearly inverted a
conclusion.** The stencil and cross-correlation predictions for LF asymmetry
correlate at -0.999, which reads as fatal collinearity. But they have opposite
signs and differ in magnitude by a factor of ~20, so fitting the wrong one
returns a slope of -27 where 1.0 would mean confusion. `cross_slope` in
experimental_design.py reports this directly: high correlation blocks the joint partials
only, and leaves the single-predictor test decisive. Both arms are strongly
discriminable at the three r values already scheduled, so the test needs no
additional training runs to reach its primary claim.

**Interior parameter values buy nothing.** Adding r between existing values
left the predictor correlation unchanged (0.9882 to 0.9878). Identifiability
comes from extending range, not densifying it. That is available on the LF arm,
where stability permits nu up to ~0.9, and unavailable on heat, where FTCS caps
r below 0.5. If the joint partials are wanted, extend nu; do not add heat runs
for this purpose.

**Seeds, not heads, are the resampling unit.** Heads within a checkpoint share
a model. `cluster_bootstrap` resamples seeds; treating heads as independent
would shrink intervals by roughly sqrt(n_heads) and make every slope
significantly different from both 0 and 1.

**Two numerical guards, both from smoke-test failures.** Predictors sitting at
1e-17 across all conditions are numerically zero, not weak signals; an absolute
tolerance kept them and the regression returned coefficients of order 1e8. The
check is now scale-aware. Separately, on symmetric arms the cross-correlation
reduces to the autocorrelation exactly, so `_near_duplicate` prunes it rather
than producing a singular design.

**offsets_full spans -N/2 .. N/2-1.** The Nyquist offset has no positive
counterpart, so any directional statistic summed over the full ring is
one-sided: the xcorr diagnostic initially reported a spurious negative centroid
for the symmetric heat arm. All attention and parameter-tracking statistics use symmetric windows via
`restrict`; anything added later must too.

**The cross-import must resolve to the attention analysis's actual filenames.** Its
pipeline scripts are `assign_heads.py`, `build_predictors.py`,
`extract_attention_profiles.py`, and `selftest.py` (plus
`attention_common.py`) in `analysis/what_do_heads_learn`.
`attention_statistics.py` and `cross_correlation.py` import `assign_heads`
and `build_predictors` directly, so those names have to match on disk; a
stray local copy of `attention_common.py` in this directory previously masked that
and had already drifted from the shared version by the time it was found.
There is deliberately no duplicate head-analysis module in this directory now—
only the `sys.path` insert at the top of `experimental_design.py`,
`attention_statistics.py`, and `cross_correlation.py` pointing at
`../what_do_heads_learn`.

## Outputs

- `stats.csv` — one row per head with the four measured statistics and all
  predictor-implied values. Joins to the attention labels on
  (pde, r, seed, layer, head).
- `stats_predicted.csv` — the condition-level prediction table, which is close
  to publication-ready as the predictions table for Section 2.
- `regression.csv` — coefficients with cluster-bootstrap CIs, VIF, residual df,
  and between-condition R2 for every model.
- `parameter_tracking.png/pdf` — measured statistic against r with both
  predicted curves overlaid, per PDE.
