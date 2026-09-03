import subprocess, os, time, shutil
import torch
#SLOTS = ["0"] 
SLOTS      = ["0", "0", "0", "1", "1"]   # 2 GPUs, 2 parallel instances each = 4 slots
R          = "0.1"                  # diffusivity for heat -- adjust per sweep value
EPOCHS     = 60                     # matches established convergence for heat (see project notes --
                                     # heat r-sweep converged cleanly at 10 epochs; change only if
                                     # you have a specific reason to retrain at higher epochs
TAG        = f"heat_new_{R}"
SEED_START = 111                    # heat seed range convention: 100-119
SEED_END   = 112
GENERATOR  = "generate_heat_equation.py"
PARAM_FLAG = "--alpha"              # confirmed flag name for this generator (heat uses --alpha, not --r)

DATA_ROOT    = f"data/heat_new_{R}"
RESULTS_ROOT = f"final_results/heat_new_{R}"
CKPT_ROOT    = f"final_checkpoints/heat_new_{R}"

os.makedirs("logs", exist_ok=True)


def checkpoint_alpha_matches(ckpt_path, expected_r, tol=1e-6):
    """
    Loads a checkpoint's saved args and checks whether its recorded alpha
    actually matches the r-value this run is currently configured for.
    Guards against silently trusting stale results from a previous run that
    used the wrong r-value but saved to the same-looking directory names.
    """
    if not os.path.exists(ckpt_path):
        return False
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        saved_alpha = ckpt.get("args", {}).get("alpha", None)
        if saved_alpha is None:
            return False
        return abs(float(saved_alpha) - float(expected_r)) < tol
    except Exception as e:
        print(f"  WARNING: couldn't verify checkpoint {ckpt_path}: {e}")
        return False


def seed_status(seed):
    """
    Returns 'done', 'metrics_only', 'resume', or 'fresh' for a given seed by
    checking what already exists on disk -- lets a rerun after an interruption
    skip completed work at whatever stage it actually stopped, rather than
    treating "training finished" and "fully analyzed" as the same thing.

    Every "done" or "metrics_only" verdict is cross-checked against the
    checkpoint's saved alpha value: a corr_matrix.npy that exists but was
    produced from a run at a DIFFERENT r-value is treated as stale, not done,
    since directory names alone don't guarantee the results inside are for
    the r-value this run is currently configured for.
    """
    results_dir = f"{RESULTS_ROOT}/seed_{seed}"
    ckpt_dir = f"{CKPT_ROOT}/seed_{seed}"
    data_dir = f"{DATA_ROOT}/seed_{seed}"
    best_ckpt = f"{ckpt_dir}/best_model.pt"

    have_corr = os.path.exists(f"{results_dir}/corr_matrix.npy")
    have_final = os.path.exists(f"{ckpt_dir}/final_model.pt")

    if have_corr or have_final:
        if not checkpoint_alpha_matches(best_ckpt, R):
            print(f"  seed {seed}: found existing results/checkpoint but alpha "
                  f"doesn't match r={R} -- deleting stale files, will rerun from scratch")
            if os.path.exists(ckpt_dir):
                shutil.rmtree(ckpt_dir)
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
            return "fresh"

    if have_corr:
        return "done"

    if have_final and os.path.exists(f"{data_dir}/test.npy"):
        return "metrics_only"

    if os.path.exists(f"{ckpt_dir}/resume_state.pt"):
        return "resume"

    return "fresh"


all_seeds = list(range(SEED_START, SEED_END))
queue = []
for seed in all_seeds:
    status = seed_status(seed)
    if status == "done":
        print(f"seed {seed}: already complete, skipping")
    else:
        queue.append((seed, status))

print(f"\n{len(queue)}/{len(all_seeds)} seeds need to run "
      f"({sum(1 for _, s in queue if s == 'resume')} resuming training, "
      f"{sum(1 for _, s in queue if s == 'metrics_only')} metrics-only, "
      f"{sum(1 for _, s in queue if s == 'fresh')} fresh)\n")

procs = {i: None for i in range(len(SLOTS))}

while queue or any(p is not None for p in procs.values()):
    for slot, p in procs.items():
        if p is not None and p.poll() is not None:
            if p.returncode != 0:
                print(f"WARNING: slot {slot} job exited with code {p.returncode} -- check its log")
            procs[slot] = None

        if procs[slot] is None and queue:
            seed, status = queue.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = SLOTS[slot]
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

            data_dir = f"{DATA_ROOT}/seed_{seed}"
            ckpt_dir = f"{CKPT_ROOT}/seed_{seed}"
            results_dir = f"{RESULTS_ROOT}/seed_{seed}"
            resume_path = f"{ckpt_dir}/resume_state.pt"

            train_cmd = (
                f"python transformer.py --data_dir {data_dir} --save_dir {ckpt_dir}"
                f" --seed {seed} --alpha {R} --epochs {EPOCHS} --batch_size 256"
            )

            if status == "metrics_only":
                # Training already finished (final_model.pt exists) -- skip
                # straight to metrics.py, no data regeneration or retraining.
                cmd = (
                    f"python analysis/metrics.py --checkpoint {ckpt_dir}/best_model.pt"
                    f" --data_dir {data_dir} --stencil_path {data_dir}/stencil.npy"
                    f" --save_dir {results_dir}"
                    f" && rmdir /s /q {data_dir.replace('/', chr(92))}"
                )
            elif status == "resume":
                # Data directory should still exist from the interrupted attempt --
                # skip regeneration, resume training, then continue the pipeline.
                train_cmd += f" --resume_from {resume_path}"
                cmd = (
                    f"{train_cmd}"
                    f" && python analysis/metrics.py --checkpoint {ckpt_dir}/best_model.pt"
                    f" --data_dir {data_dir} --stencil_path {data_dir}/stencil.npy"
                    f" --save_dir {results_dir}"
                    f" && rmdir /s /q {data_dir.replace('/', chr(92))}"
                )
            else:
                cmd = (
                    f"python {GENERATOR} {PARAM_FLAG} {R} --seed {seed} --save_dir {data_dir}"
                    f" && {train_cmd}"
                    f" && python analysis/metrics.py --checkpoint {ckpt_dir}/best_model.pt"
                    f" --data_dir {data_dir} --stencil_path {data_dir}/stencil.npy"
                    f" --save_dir {results_dir}"
                    f" && rmdir /s /q {data_dir.replace('/', chr(92))}"
                )

            log = open(f"logs/{TAG}_seed{seed}.log", "w")
            procs[slot] = subprocess.Popen(cmd, shell=True, env=env,
                                            stdout=log, stderr=subprocess.STDOUT)
            print(f"slot {slot} (gpu {SLOTS[slot]}): seed {seed} ({status})")

    time.sleep(5)

print("done")