# Attention-only positive-control protocol

Status before the full run: frozen implementation protocol; numerical results
are not assumed.

## Question

Can the existing diagnostics recognize the known finite-difference update when
attention is the only operation that mixes spatial positions?

## Model

For offsets `[-1, 0, +1]`, the model predicts

\[
\widehat u^{t+1}_i = \sum_d \operatorname{softmax}(\theta)_d u^t_{i+d}.
\]

There is one attention row, identity value and output maps, no residual path,
no feed-forward block, no normalization layer, and no other spatial operation.
The analytical coefficients are not supplied to the fitter.

## Data and uncertainty

- Same six heat and Lax--Friedrichs conditions as the ordinary transformers.
- Same 20 generated-data seeds per condition: heat 100--119 and LF 300--319.
- Existing train, validation, and test files are read without modification.
- All training transitions enter the empirical-risk objective once.
- Validation and test use deterministic 2,048-pair samples per seed.
- Model/data seed is the uncertainty unit; 20,000 seed-bootstrap resamples are
  used within each condition.

## Predeclared diagnostics

1. Apply the unchanged attention-analysis entropy gate, baseline-calibrated fit floor, Jensen--
   Shannon distance, and spectral-margin rule.
2. Regress heat centrality and LF asymmetry against their analytical values;
   the diagnostic target is intercept zero and slope one.
3. Compare fitted attention, the exact fixed-attention no-op, initialization,
   persistence, and analytical-stencil replacement on held-out test pairs.
4. Preserve condition-specific results. Do not pool a failed condition into a
   global claim of successful detectability.

## Interpretation gate

A condition is described as fully detected only when all 20 seeds receive the
unchanged stencil label. If the constrained model predicts accurately but
the diagnostic does not recognize its attention row, the result is a
diagnostic limitation rather than validation of the ordinary-transformer
conclusion.

## Environment change

The project environment remains on PyTorch 2.7.1 but uses the official CUDA
12.8 wheel so `sm_120` is available for the RTX 5060 Ti. Rollback command:

```powershell
& .\.venv\Scripts\python.exe -B -m pip install --force-reinstall `
  --no-deps torch==2.7.1 `
  --index-url https://download.pytorch.org/whl/cu118
```
