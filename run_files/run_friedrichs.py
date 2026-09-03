import subprocess, os, time

SLOTS      = ["0", "0", "0", "1", "1"]   # 2 GPUs, 2 parallel instances each = 4 slots
R          = "0.4"                  # Courant number for lax_friedrichs -- adjust per sweep value
EPOCHS     = 65                      # matches established convergence for lax_friedrichs (see project notes);
                                     # 65 in the original draft had no supporting evidence -- change back
                                     # only if you have a specific reason to retrain at higher epochs
TAG        = f"lf_r{R}"
SEED_START = 315                    # lax_friedrichs seed range convention: 300-319
SEED_END   = 320
GENERATOR  = "generate_lax_friedrichs.py"
PARAM_FLAG = "--r"                  # confirmed flag name for this generator

DATA_ROOT    = f"data/lax_friedrichs_r_{R}"
RESULTS_ROOT = f"final_results/lax_friedrichs_r_{R}"
CKPT_ROOT    = f"final_checkpoints/lax_friedrichs_r_{R}"

os.makedirs("logs", exist_ok=True)

jobs  = list(range(SEED_START, SEED_END))
procs = {i: None for i in range(len(SLOTS))}
queue = list(jobs)

while queue or any(p is not None for p in procs.values()):
    for slot, p in procs.items():
        if p is not None and p.poll() is not None:
            if p.returncode != 0:
                print(f"WARNING: slot {slot} job exited with code {p.returncode} -- check its log")
            procs[slot] = None

        if procs[slot] is None and queue:
            seed = queue.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = SLOTS[slot]
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

            data_dir = f"{DATA_ROOT}/seed_{seed}"
            ckpt_dir = f"{CKPT_ROOT}/seed_{seed}"
            results_dir = f"{RESULTS_ROOT}/seed_{seed}"

            # Full pipeline per job: generate -> train -> metrics -> cleanup data.
            # Chained with && so a failed step stops the rest for that seed
            # (matches the batch-script behavior used elsewhere in this project).
            cmd = (
                f"python {GENERATOR} {PARAM_FLAG} {R} --seed {seed} --save_dir {data_dir}"
                f" && python transformer.py --data_dir {data_dir} --save_dir {ckpt_dir}"
                f" --seed {seed} --alpha {R} --epochs {EPOCHS} --batch_size 256"
                f" && python analysis/metrics.py --checkpoint {ckpt_dir}/best_model.pt"
                f" --data_dir {data_dir} --stencil_path {data_dir}/stencil.npy"
                f" --save_dir {results_dir}"
                f" && rmdir /s /q {data_dir.replace('/', chr(92))}"
            )

            log = open(f"logs/{TAG}_seed{seed}.log", "w")
            procs[slot] = subprocess.Popen(cmd, shell=True, env=env,
                                            stdout=log, stderr=subprocess.STDOUT)
            print(f"slot {slot} (gpu {SLOTS[slot]}): seed {seed}")

    time.sleep(5)

print("done")