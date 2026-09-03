### Detectability control

With attention as the only spatial mixing operation, the unchanged diagnostic assigned 120/120 runs to the stencil and recovered its coefficients (maximum absolute error 1.25e-03). Tracking slopes were 1.000 for heat and 1.001 for Lax--Friedrichs, versus 0.000 and -0.450 in the trained transformers. Replacing learned attention with the exact stencil changes the control result by -1.11e-07 to -5.42e-12 on the reported scale, compared with 2.26 to 29.64 in the trained transformers. Fixing each head at its validation-set mean gives 0.013 to 0.195.

**Figure X. Detectability control.** (a) Stencil centrality or asymmetry versus attention centrality or asymmetry. Points are condition means and bars are 95% model-seed bootstrap intervals. (b) Increase in test MSE after fixing attention or replacing it with the exact stencil. The trained-transformer The attention analysis assigned 0 heads to the stencil.
