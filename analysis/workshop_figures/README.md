# NeurIPS workshop figures

These scripts regenerate the four main workshop figures from existing
attention, parameter-tracking, intervention, and Fourier result artifacts.
They do not train models, modify checkpoints, or
overwrite the original analysis outputs.

Run all figures from the repository root:

```powershell
python analysis/workshop_figures/make_all.py
```

Outputs are written as PDF and PNG files under
`analysis/workshop_figures/output/`.

Each script can also be run independently:

```powershell
python analysis/workshop_figures/figure1_task_performance.py
python analysis/workshop_figures/figure2_attention.py
python analysis/workshop_figures/figure3_substitutions.py
python analysis/workshop_figures/figure4_operator.py
```

## Corrected robustness figures

The review controls use separate scripts so that the earlier figures remain
unchanged:

```powershell
python analysis/workshop_figures/figure3_substitutions_robust.py
python analysis/workshop_figures/figure4_operator_robust.py
```

Their PDF and PNG outputs are written to
`analysis/workshop_figures/output_robustness/`. Figure 3 uses validation-
derived, head-specific fixed attention and model-seed bootstrap intervals.
Figure 4 uses the paired Fourier delta and reports the circulant residual after
subtracting identity from the Jacobian. Both scripts use the same opt-in
NeurIPS stylesheet as the original figure set.
