# Package runners and historical launcher snapshots

`run_checks.ps1`, `make_figures.ps1`, `rebuild_derived_results.ps1`, and
`refresh_from_parent.ps1` are package utilities and are invoked from the
package root as documented in the top-level README.

`run_heat.py` and `run_friedrichs.py` record the local GPU scheduling and
training settings used in the parent project. Do not run those launchers
directly. They contain hard-coded GPU assignments, mutable seed
ranges, machine-specific paths, and cleanup commands that can remove generated
data after a run. Use the individual generator, training, and metric scripts
listed in the top-level README when constructing a portable workflow.
