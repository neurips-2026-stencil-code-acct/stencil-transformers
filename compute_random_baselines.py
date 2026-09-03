"""
compute_random_baselines.py

Properly-powered random (untrained) baseline for the attention-vs-stencil correlation
metric in metrics.py. The random baseline inside metrics.py itself uses only 1 seed and
256 examples -- fine as a quick sanity check, badly underpowered as an actual statistical
baseline (your trained models are evaluated on up to n_batches*batch_size = 51,200 examples).

This script averages over --n_seeds (default 30) independent untrained initializations,
each evaluated on the same number of examples as a trained-model run, and optionally runs
a Mann-Whitney U test against one or more trained corr_matrix.npy files already saved by
metrics.py.

Model code (SpatialEmbedding, VanillaTransformer) and metric functions (extract_nb, corr)
are copied verbatim from metrics.py to guarantee identical behavior. If you change
metrics.py, mirror the change here.

Usage:
    # Baseline only:
    python compute_random_baselines.py --data_dir data\\heat_nlayers_4\\seed_100 \\
        --stencil_path data\\heat_nlayers_4\\seed_100\\stencil.npy \\
        --n_layers 4 --n_heads 4 --n_seeds 30 --n_batches 200 --batch_size 256 \\
        --save_dir results\\random_baseline_nlayers_4

    # With comparison against one or more trained runs' corr_matrix.npy:
    python compute_random_baselines.py --data_dir data\\heat_nlayers_4\\seed_100 \\
        --stencil_path data\\heat_nlayers_4\\seed_100\\stencil.npy \\
        --n_layers 4 --n_heads 4 --n_seeds 30 \\
        --trained_glob "results\\heat_nlayers_4\\seed_*\\corr_matrix.npy" \\
        --save_dir results\\random_baseline_nlayers_4
"""

import os
import glob
import sys
import argparse
import subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, mannwhitneyu

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=None,
                     help="Required unless --load_baseline is set")
parser.add_argument("--data_file", type=str, default="test.npy",
                     help="Which file in data_dir to draw samples from (metrics.py uses test.npy)")
parser.add_argument("--stencil_path", type=str, default=None,
                     help="Required unless --load_baseline is set")
parser.add_argument("--save_dir", type=str, default="results/random_baseline")

parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--d_ff", type=int, default=256)

parser.add_argument("--n_seeds", type=int, default=30,
                     help="Number of independent untrained initializations")
parser.add_argument("--n_batches", type=int, default=200,
                     help="Batches of attention to collect per seed (matched to metrics.py's default)")
parser.add_argument("--batch_size", type=int, default=256)

parser.add_argument("--trained_glob", type=str, default=None,
                     help="Optional glob pattern matching one or more trained corr_matrix.npy "
                          "files (e.g. from multiple seeds) to run a Mann-Whitney U test against "
                          "the random baseline")

parser.add_argument("--regenerate_data", action="store_true",
                     help="If test.npy is missing from --data_dir (e.g. because the batch script "
                          "rmdir'd it after training), call generate_heat_equation.py to recreate "
                          "it before running the baseline. Requires --alpha and --gen_seed, and "
                          "assumes generate_heat_equation.py is on PATH / in the current directory.")
parser.add_argument("--alpha", type=float, default=None,
                     help="Diffusivity value to pass to generate_heat_equation.py if regenerating data")
parser.add_argument("--gen_seed", type=int, default=None,
                     help="Seed to pass to generate_heat_equation.py if regenerating data "
                          "(this is the ORIGINAL training seed for this condition, not a baseline seed 0..n_seeds)")
parser.add_argument("--generator_script", type=str, default="generate_heat_equation.py",
                     help="Path to the data-generation script to call if regenerating")
parser.add_argument("--param_flag_name", type=str, default="--alpha",
                     help="Name of the CLI flag the generator script expects for the diffusivity/CFL "
                          "parameter -- e.g. '--alpha' for generate_heat_equation.py, but possibly "
                          "'--r' for other generators (e.g. generate_lax_friedrichs.py). Check the "
                          "target script's argparse before running -- a mismatch here causes the "
                          "subprocess call to fail with an 'unrecognized arguments' error.")
parser.add_argument("--load_baseline", type=str, default=None,
                     help="Path to a previously-saved random_baseline_corrs.npy. If set, skips "
                          "regenerating the 30 untrained models entirely and just reruns the "
                          "statistical comparison against --trained_glob using this saved baseline.")
parser.add_argument("--load_baseline_attn", type=str, default=None,
                     help="Path to a previously-saved random_baseline_mean_attn.npy (shape "
                          "[n_seeds, n_layers, n_heads, NX, NX]). If set, recomputes corr AND "
                          "locality from these saved attention matrices without rerunning any "
                          "models or reloading data -- requires --stencil_path.")
parser.add_argument("--trained_locality_glob", type=str, default=None,
                     help="Optional glob pattern matching trained locality_matrix.npy files "
                          "(e.g. 'results\\\\heat_nlayers_4\\\\seed_*\\\\locality_matrix.npy') to run "
                          "a Mann-Whitney U test on locality, trained vs. random baseline.")
parser.add_argument("--trained_entropy_glob", type=str, default=None,
                     help="Optional glob pattern matching trained entropy_matrix.npy files "
                          "to run a Mann-Whitney U test on entropy, trained vs. random baseline.")
args = parser.parse_args()

if not args.load_baseline and not args.load_baseline_attn and (args.data_dir is None or args.stencil_path is None):
    parser.error("--data_dir and --stencil_path are required unless --load_baseline or --load_baseline_attn is set")
if args.load_baseline_attn and args.stencil_path is None:
    parser.error("--load_baseline_attn requires --stencil_path (to recompute corr) even though no training data is needed")

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

NX = 64  # matches metrics.py's hardcoded value


# ============ Model (verbatim from metrics.py) ============
class SpatialEmbedding(nn.Module):
    def __init__(self, nx, d_model):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        pe = torch.zeros(nx, d_model)
        pos = torch.arange(nx).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.value_proj(x)
        return x + self.pe


class VanillaTransformer(nn.Module):
    def __init__(self, nx, n_layers, n_heads, d_model, d_ff, dropout):
        super().__init__()
        self.embedding = SpatialEmbedding(nx, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output_head(x).squeeze(-1)


# ============ Attention collection (memory-optimized: running sum, not stored batches) ============
def collect_attn_mean(mdl, loader, n_batches, device, n_layers, n_heads, nx):
    """
    Returns the MEAN attention matrix per (layer, head), accumulated as a running
    sum during the forward pass rather than storing every individual example's
    attention matrix. This keeps peak memory at O(n_layers * n_heads * nx * nx)
    regardless of how many batches/examples are processed, instead of
    O(n_layers * n_batches * batch_size * n_heads * nx * nx) -- the latter is what
    caused out-of-memory on large n_batches (e.g. ~12.5GB for n_batches=200,
    batch_size=256, 4 layers -- easily exceeds 16GB RAM).
    """
    sums = {i: torch.zeros(n_heads, nx, nx) for i in range(n_layers)}
    counts = {i: 0 for i in range(n_layers)}

    def make_hook(idx):
        def hook(module, inp, out):
            with torch.no_grad():
                # norm_first=True: self_attn actually runs on norm1(src) during
                # the real forward pass, not on the raw layer input.
                normed = module.norm1(inp[0])
                _, w = module.self_attn(normed, normed, normed,
                    need_weights=True, average_attn_weights=False)
                w = w.detach().cpu()  # (B, H, nx, nx)
                sums[idx] += w.sum(dim=0)
                counts[idx] += w.shape[0]
        return hook

    hooks = [layer.register_forward_hook(make_hook(i))
             for i, layer in enumerate(mdl.transformer.layers)]
    mdl.eval()
    with torch.no_grad():
        for bi, x in enumerate(loader):
            if bi >= n_batches:
                break
            _ = mdl(x.to(device))
    for h in hooks:
        h.remove()

    return {i: (sums[i] / counts[i]).numpy() if counts[i] > 0 else None
            for i in range(n_layers)}


# ============ Metric functions (verbatim from metrics.py) ============
def extract_nb(attn, nx):
    nb = np.zeros((nx, 3))
    for i in range(nx):
        nb[i] = [attn[i, (i - 1) % nx], attn[i, i], attn[i, (i + 1) % nx]]
    return nb


def corr(attn, stencil, nx):
    mean = extract_nb(attn, nx).mean(0)
    if np.std(mean) < 1e-10 or np.std(stencil) < 1e-10:
        return 0.0
    return float(pearsonr(mean, stencil)[0])


def locality(attn, nx):
    return float(extract_nb(attn, nx).sum(1).mean())


def entropy(attn):
    a = np.clip(attn, 1e-10, 1.0)
    return float(-(a * np.log(a)).sum(1).mean())


def ensure_data_available(data_dir, data_file):
    data_path = os.path.join(data_dir, data_file)
    if os.path.exists(data_path):
        return

    if not args.regenerate_data:
        raise FileNotFoundError(
            f"{data_path} not found. If it was deleted by the training batch script's "
            f"`rmdir /s /q`, rerun with --regenerate_data --alpha <value> --gen_seed <original seed> "
            f"to recreate it via {args.generator_script}."
        )

    if args.alpha is None or args.gen_seed is None:
        raise ValueError(
            "--regenerate_data requires --alpha and --gen_seed to be set "
            "(the original diffusivity and seed used to train this condition)."
        )

    print(f"{data_path} missing -- regenerating via {args.generator_script} "
          f"({args.param_flag_name}={args.alpha}, seed={args.gen_seed})...")
    cmd = [
        sys.executable, args.generator_script,
        args.param_flag_name, str(args.alpha),
        "--seed", str(args.gen_seed),
        "--save_dir", data_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"{args.generator_script} failed with exit code {result.returncode}")
    print(f"Regenerated data at {data_dir}")


def main():
    all_L = None  # locality matrix, populated in either branch below
    all_E = None  # entropy matrix, populated in either branch below

    if args.load_baseline_attn:
        print(f"Loading saved attention matrices from {args.load_baseline_attn} (skipping model runs entirely)")
        all_mean_attn = np.load(args.load_baseline_attn)  # (n_seeds, n_layers, n_heads, NX, NX)
        stencil = np.load(args.stencil_path)
        n_seeds, n_layers, n_heads, nx, _ = all_mean_attn.shape

        all_C = np.zeros((n_seeds, n_layers, n_heads))
        all_L = np.zeros((n_seeds, n_layers, n_heads))
        all_E = np.zeros((n_seeds, n_layers, n_heads))
        for s in range(n_seeds):
            for li in range(n_layers):
                for hi in range(n_heads):
                    ma = all_mean_attn[s, li, hi]
                    all_C[s, li, hi] = corr(ma, stencil, nx)
                    all_L[s, li, hi] = locality(ma, nx)
                    all_E[s, li, hi] = entropy(ma)
        flat = all_C.flatten()
        print(f"Recomputed corr + locality + entropy from saved attention: shape={all_C.shape}, "
              f"corr mean={flat.mean():.4f} +/- {flat.std():.4f}, "
              f"locality mean={all_L.mean():.4f} +/- {all_L.std():.4f}, "
              f"entropy mean={all_E.mean():.4f} +/- {all_E.std():.4f}")

    elif args.load_baseline:
        print(f"Loading saved baseline from {args.load_baseline} (skipping regeneration)")
        all_C = np.load(args.load_baseline)
        flat = all_C.flatten()
        print(f"Loaded baseline: shape={all_C.shape}, mean={flat.mean():.4f} +/- {flat.std():.4f}")

    else:
        ensure_data_available(args.data_dir, args.data_file)

        stencil = np.load(args.stencil_path)
        print(f"Stencil: {stencil}")

        data = np.load(os.path.join(args.data_dir, args.data_file))
        inputs = torch.tensor(data[:, :-1, :].reshape(-1, NX), dtype=torch.float32)
        loader = DataLoader(inputs, batch_size=args.batch_size, shuffle=False)

        n_available = inputs.shape[0]
        n_requested = args.n_batches * args.batch_size
        if n_requested > n_available:
            print(f"Warning: requested {n_requested} examples but only {n_available} available; "
                  f"will iterate the full dataset once per seed instead.")

        all_C = np.zeros((args.n_seeds, args.n_layers, args.n_heads))
        all_L = np.zeros((args.n_seeds, args.n_layers, args.n_heads))
        all_E = np.zeros((args.n_seeds, args.n_layers, args.n_heads))
        all_mean_attn = np.zeros((args.n_seeds, args.n_layers, args.n_heads, NX, NX), dtype=np.float32)

        for seed in range(args.n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = VanillaTransformer(NX, args.n_layers, args.n_heads,
                                        args.d_model, args.d_ff, dropout=0.0).to(device)
            model.eval()

            attn_means = collect_attn_mean(model, loader, args.n_batches, device, args.n_layers, args.n_heads, NX)

            C = np.zeros((args.n_layers, args.n_heads))
            L = np.zeros((args.n_layers, args.n_heads))
            E = np.zeros((args.n_layers, args.n_heads))
            for li in range(args.n_layers):
                if attn_means[li] is None:
                    continue
                for hi in range(args.n_heads):
                    ma = attn_means[li][hi]
                    C[li, hi] = corr(ma, stencil, NX)
                    L[li, hi] = locality(ma, NX)
                    E[li, hi] = entropy(ma)
                    all_mean_attn[seed, li, hi] = ma

            all_C[seed] = C
            all_L[seed] = L
            all_E[seed] = E
            print(f"Seed {seed}: mean corr = {C.mean():.4f} +/- {C.std():.4f}, "
                  f"mean locality = {L.mean():.4f}, mean entropy = {E.mean():.4f}")

        np.save(os.path.join(args.save_dir, "random_baseline_corrs.npy"), all_C)
        np.save(os.path.join(args.save_dir, "random_baseline_locality.npy"), all_L)
        np.save(os.path.join(args.save_dir, "random_baseline_entropy.npy"), all_E)
        np.save(os.path.join(args.save_dir, "random_baseline_mean_attn.npy"), all_mean_attn)
        print(f"Saved raw averaged attention matrices to {args.save_dir}/random_baseline_mean_attn.npy "
              f"(shape: {all_mean_attn.shape} = [seed, layer, head, NX, NX]) -- "
              f"any future metric (entropy, L2, KL, etc.) can be recomputed from this without rerunning models.")

        flat = all_C.flatten()
        print("\n" + "=" * 50)
        print(f"Random baseline (n_seeds={args.n_seeds}, n_examples/seed={min(n_requested, n_available)}):")
        print(f"  Mean corr: {flat.mean():.4f} +/- {flat.std():.4f}")
        print(f"  Per-seed mean range: [{all_C.mean(axis=(1,2)).min():.4f}, {all_C.mean(axis=(1,2)).max():.4f}]")
        print("=" * 50)

        print(f"Saved raw baseline correlations to {args.save_dir}/random_baseline_corrs.npy")

    if args.trained_glob:
        files = sorted(glob.glob(args.trained_glob))
        if not files:
            print(f"\nWarning: no files matched --trained_glob '{args.trained_glob}'")
        else:
            trained_stack = np.stack([np.load(f) for f in files])  # (n_trained_seeds, n_layers, n_heads)
            trained_flat = trained_stack.flatten()

            # --- Signed correlation (original) ---
            stat_signed, p_signed = mannwhitneyu(trained_flat, flat, alternative="two-sided")
            print(f"\nMann-Whitney U test (SIGNED corr): trained (n={len(trained_flat)}, {len(files)} seeds) "
                  f"vs. random baseline (n={len(flat)}, {args.n_seeds} seeds)")
            print(f"  Trained mean: {trained_flat.mean():.4f} +/- {trained_flat.std():.4f}")
            print(f"  Baseline mean: {flat.mean():.4f} +/- {flat.std():.4f}")
            print(f"  U statistic: {stat_signed:.2f}")
            print(f"  p-value: {p_signed:.6e}")

            # --- abs(correlation) -- avoids sign-cancellation washing out real structure ---
            trained_abs = np.abs(trained_flat)
            baseline_abs = np.abs(flat)
            stat_abs, p_abs = mannwhitneyu(trained_abs, baseline_abs, alternative="two-sided")
            print(f"\nMann-Whitney U test (ABS corr): trained (n={len(trained_abs)}) "
                  f"vs. random baseline (n={len(baseline_abs)})")
            print(f"  Trained mean(abs): {trained_abs.mean():.4f} +/- {trained_abs.std():.4f}")
            print(f"  Baseline mean(abs): {baseline_abs.mean():.4f} +/- {baseline_abs.std():.4f}")
            print(f"  U statistic: {stat_abs:.2f}")
            print(f"  p-value: {p_abs:.6e}")

            if p_signed >= 0.05 and p_abs < 0.05:
                print(f"\n  NOTE: signed correlation was not significant (p={p_signed:.4f}) but abs(corr) IS "
                      f"(p={p_abs:.4f}) -- consistent with sign-cancellation across heads masking real "
                      f"shape-level structure. Consider reporting abs(corr) as the shape-level test.")

            np.savez(os.path.join(args.save_dir, "mannwhitney_result.npz"),
                     trained_mean=trained_flat.mean(), trained_std=trained_flat.std(),
                     baseline_mean=flat.mean(), baseline_std=flat.std(),
                     U_statistic_signed=stat_signed, p_value_signed=p_signed,
                     trained_mean_abs=trained_abs.mean(), trained_std_abs=trained_abs.std(),
                     baseline_mean_abs=baseline_abs.mean(), baseline_std_abs=baseline_abs.std(),
                     U_statistic_abs=stat_abs, p_value_abs=p_abs,
                     n_trained=len(trained_flat), n_baseline=len(flat))
            print(f"\nSaved comparison to {args.save_dir}/mannwhitney_result.npz")

    if args.trained_locality_glob:
        if all_L is None:
            print(f"\nWarning: --trained_locality_glob given but no locality baseline available "
                  f"(rerun with --load_baseline_attn or regenerate the baseline to get locality data).")
        else:
            files = sorted(glob.glob(args.trained_locality_glob))
            if not files:
                print(f"\nWarning: no files matched --trained_locality_glob '{args.trained_locality_glob}'")
            else:
                trained_L_stack = np.stack([np.load(f) for f in files])
                trained_L_flat = trained_L_stack.flatten()
                baseline_L_flat = all_L.flatten()

                stat_loc, p_loc = mannwhitneyu(trained_L_flat, baseline_L_flat, alternative="two-sided")
                print(f"\nMann-Whitney U test (LOCALITY): trained (n={len(trained_L_flat)}, {len(files)} seeds) "
                      f"vs. random baseline (n={len(baseline_L_flat)})")
                print(f"  Trained mean locality: {trained_L_flat.mean():.4f} +/- {trained_L_flat.std():.4f}")
                print(f"  Baseline mean locality: {baseline_L_flat.mean():.4f} +/- {baseline_L_flat.std():.4f}")
                print(f"  U statistic: {stat_loc:.2f}")
                print(f"  p-value: {p_loc:.6e}")

                np.savez(os.path.join(args.save_dir, "mannwhitney_locality_result.npz"),
                         trained_mean=trained_L_flat.mean(), trained_std=trained_L_flat.std(),
                         baseline_mean=baseline_L_flat.mean(), baseline_std=baseline_L_flat.std(),
                         U_statistic=stat_loc, p_value=p_loc,
                         n_trained=len(trained_L_flat), n_baseline=len(baseline_L_flat))
                print(f"Saved locality comparison to {args.save_dir}/mannwhitney_locality_result.npz")

    if args.trained_entropy_glob:
        if all_E is None:
            print(f"\nWarning: --trained_entropy_glob given but no entropy baseline available "
                  f"(rerun with --load_baseline_attn or regenerate the baseline to get entropy data).")
        else:
            files = sorted(glob.glob(args.trained_entropy_glob))
            if not files:
                print(f"\nWarning: no files matched --trained_entropy_glob '{args.trained_entropy_glob}'")
            else:
                trained_E_stack = np.stack([np.load(f) for f in files])
                trained_E_flat = trained_E_stack.flatten()
                baseline_E_flat = all_E.flatten()

                stat_ent, p_ent = mannwhitneyu(trained_E_flat, baseline_E_flat, alternative="two-sided")
                print(f"\nMann-Whitney U test (ENTROPY): trained (n={len(trained_E_flat)}, {len(files)} seeds) "
                      f"vs. random baseline (n={len(baseline_E_flat)})")
                print(f"  Trained mean entropy: {trained_E_flat.mean():.4f} +/- {trained_E_flat.std():.4f}")
                print(f"  Baseline mean entropy: {baseline_E_flat.mean():.4f} +/- {baseline_E_flat.std():.4f}")
                print(f"  U statistic: {stat_ent:.2f}")
                print(f"  p-value: {p_ent:.6e}")

                np.savez(os.path.join(args.save_dir, "mannwhitney_entropy_result.npz"),
                         trained_mean=trained_E_flat.mean(), trained_std=trained_E_flat.std(),
                         baseline_mean=baseline_E_flat.mean(), baseline_std=baseline_E_flat.std(),
                         U_statistic=stat_ent, p_value=p_ent,
                         n_trained=len(trained_E_flat), n_baseline=len(baseline_E_flat))
                print(f"Saved entropy comparison to {args.save_dir}/mannwhitney_entropy_result.npz")


if __name__ == "__main__":
    main()