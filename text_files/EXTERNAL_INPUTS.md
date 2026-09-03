# External data and checkpoint inputs

The snapshot is self-contained for inspecting saved results, rerunning tests,
recomputing robustness summaries, and rebuilding figures. Repeating model
training or checkpoint-level extraction requires two large directories in the
parent project.

## Expected locations

```text
stencil-transformers/
├── data/
│   ├── heat_new_0.1/seed_100/{train,val,test,stencil}.npy
│   ├── heat_new_0.25/seed_100/{train,val,test,stencil}.npy
│   ├── heat_new_0.4/seed_100/{train,val,test,stencil}.npy
│   ├── lax_friedrichs_r_0.1/seed_300/{train,val,test,stencil}.npy
│   ├── lax_friedrichs_r_0.25/seed_300/{train,val,test,stencil}.npy
│   └── lax_friedrichs_r_0.4/seed_300/{train,val,test,stencil}.npy
├── final_checkpoints/
│   ├── heat_new_0.1/seed_100/best_model.pt
│   ├── ...
│   └── lax_friedrichs_r_0.4/seed_319/best_model.pt
└── paper_reproducibility/
```

The full experiment uses heat seeds 100--119 and Lax--Friedrichs seeds
300--319 at each of the three parameter values. Some checkpoint analyses can
use another held-out seed from the same equation/parameter condition when a
matching per-seed data directory is unavailable; the analysis code emits a
warning when it does so.

## Why they are not copied

At snapshot creation time, `data/` contained 720 files totaling about 92.2 GB,
and `final_checkpoints/` contained 1,000 files totaling about 650 MB. Copying
them would make this readable code package needlessly large and would create a
second mutable copy of the evidence.

The placeholder directories in this folder contain no links and no hidden
data. Scripts that need the large inputs should be given `..\data` and
`..\final_checkpoints` explicitly.

## Full-compute boundary

The following actions require the external inputs and substantial compute:

- retraining ordinary transformers;
- extracting all trained attention profiles;
- rerunning the head-preserving attention replacements;
- recomputing all full-model Jacobians and finite-sinusoid probes;
- refitting all 120 positive-control models.

The included `rebuild_derived_results.ps1` intentionally starts after those
steps. It recomputes only seed-level uncertainty summaries and paper exports
from the compact files already in this folder.
