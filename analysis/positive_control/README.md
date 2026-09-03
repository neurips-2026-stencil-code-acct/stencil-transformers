# S1 attention-only positive control

This folder fits the constrained S1 model, applies the unchanged attention,
parameter-tracking, and intervention
logic, validates every saved result, and builds the S3 paper artifacts. It does
not modify the ordinary-transformer checkpoints or generated data.

## Full visible run

From PowerShell at the repository root:

```powershell
& .\analysis\positive_control\run_all.ps1 -Devices "cuda:0,cuda:1" -Resume
```

The Python runner keeps one worker per GPU, writes one atomic JSON artifact per
seed, and prints a live overall progress bar with elapsed time, ETA, per-device
completion counts, and the most recent coefficient error.

## Declared preflight

```powershell
& .\.venv\Scripts\python.exe .\analysis\positive_control\selftest.py

& .\.venv\Scripts\python.exe .\analysis\positive_control\run.py `
  --conditions "heat:0.25,lf:0.25" `
  --seeds "100,300" `
  --devices "cuda:0,cuda:1" `
  --max-trajectories 8 `
  --max-time-steps 64 `
  --evaluation-pairs 256 `
  --bootstrap 1000 `
  --expected-models 2 `
  --out analysis/positive_control/smoke
```

Smoke outputs are pipeline checks, not paper evidence.
