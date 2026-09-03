# Isolated robustness checks for the workshop analysis

These scripts read existing intervention, Jacobian, and Fourier outputs and write new derived files under
`analysis/robustness_checks/results`. They do not change training code,
checkpoints, or the original analysis outputs. No retraining is required.

## What is resampled

`bootstrap_model_effects.py` uses the model initialization as the independent
unit. Within each PDE--parameter condition, it samples the 20 model seeds with
replacement. Each substitution effect is computed within a model before
resampling, so the substituted and intact measurements remain paired. For the
Fourier comparison, each seed's mode-level errors are already combined into an
RMSE for the stated frequency range; the script computes

\[
\Delta = \operatorname{RMSE}_{\mathrm{identity}}
       - \operatorname{RMSE}_{\mathrm{PDE}}
\]

before resampling seeds. Positive \(\Delta\) means that the known PDE response
is closer to the model response. Negative \(\Delta\) means identity is closer.
Heads, Fourier modes, and individual test inputs are not treated as independent
replicates. Intervals are deterministic 95% percentile-bootstrap intervals
with 20,000 resamples and master seed 20,260,824
\citep{efron1979}.

For substitutions, the primary estimate is the mean paired log MSE ratio. Its
exponential is reported as a geometric mean MSE ratio, along with the same
interval transformed back to the ordinary ratio scale. A ratio above one means
that the replacement increased prediction error.

The head-preserving analysis uses three specified within-model comparisons:

1. the validation-derived fixed matrix for each head versus intact attention;
2. one validation-derived layer mean shared by all heads versus the separate
   fixed matrices; and
3. the analytical stencil versus the separate fixed matrices.

The fixed matrices are estimated only from validation inputs and evaluated on
different test inputs. The log ratio is the primary scale because model MSE is
strictly positive and varies substantially across seeds. Raw MSE differences
and excess-error differences are reported separately so that a large ratio
caused by a very small intact-model error is not mistaken for a persistence-
scale loss.

## Why saved Jacobian outputs are enough for J-I

Let \(P(J)\) be the circular-diagonal average of \(J\), represented as a
circulant matrix. Circular-diagonal averaging is the Frobenius-orthogonal
projection onto the space of circulant matrices. The existing Jacobian files save
both \(P(J)\) and

\[
r=\frac{\lVert J-P(J)\rVert_F}{\lVert J\rVert_F}.
\]

Because identity is circulant and \(P\) is linear,

\[
(J-I)-P(J-I)=J-P(J).
\]

Subtracting identity therefore leaves the noncirculant numerator unchanged.
Orthogonality gives the Pythagorean identity

\[
\lVert J\rVert_F^2
=
\lVert P(J)\rVert_F^2
+
\lVert J-P(J)\rVert_F^2,
\]

so the saved relative residual reconstructs

\[
\lVert J\rVert_F^2
=
\frac{\lVert P(J)\rVert_F^2}{1-r^2}.
\]

The saved circular profile also gives \(\lVert P(J)-I\rVert_F\). Thus

\[
\lVert J-I\rVert_F^2
=
\lVert P(J)-I\rVert_F^2
+
\lVert J-P(J)\rVert_F^2,
\]

which determines the circulant residual of \(J-I\) without recomputing any
Jacobians. `test_robustness_checks.py` compares this reconstruction with a
direct matrix calculation and verifies that the noncirculant numerator is
unchanged.

Two edge cases are handled explicitly. If \(P(J)=0\) and \(r=1\), the saved
relative residual does not determine the absolute matrix norm. If \(J=I\),
then \(J-I=0\), and a relative residual with a zero denominator is undefined.
Neither case occurs in the current saved workshop results.

## Commands

From the repository root:

```cmd
C:\Python312\python.exe analysis\robustness_checks\test_robustness_checks.py

C:\Python312\python.exe analysis\robustness_checks\bootstrap_model_effects.py

C:\Python312\python.exe analysis\robustness_checks\j_minus_identity_circulance.py
```

The default PDE scope is `heat,lf`, matching the workshop manuscript. Pass
`--pdes heat,lf,wave` only for a separate full-scope audit.

## Outputs

`results/paired_bootstrap/` contains seed-level paired effects and condition
summaries for the intervention and Fourier analyses. `results/j_minus_identity/` contains per-seed and
condition-level circulant diagnostics for the learned update.

The validation-derived attention analysis writes:

- `head_preserving_seed_contrasts.csv`;
- `head_preserving_bootstrap_summary.csv`; and
- `head_preserving_bootstrap_metadata.json`.

The bootstrap method should cite `efron1979`; the verified BibTeX entry is in
`references.bib`. The \(J-I\) reconstruction uses linearity, orthogonal
projection, and the Pythagorean identity, so it does not introduce an
additional empirical method citation.
