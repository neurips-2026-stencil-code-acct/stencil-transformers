"""Extract per-head relative-offset attention profiles from checkpoints.

Writes one .npz per run containing the full offset profile for every
(layer, head), plus metadata. Runs on CPU by default and loads one checkpoint
at a time, so it does not contend with training for GPU memory (this is the
same constraint that forced metrics.py out of the inline sweep).

Usage
-----
    python extract_attention_profiles.py --runs "runs/**/best.pt" --dry-run
    python extract_attention_profiles.py --runs "runs/**/best.pt" \
        --data-root data --out profiles/trained
    python extract_attention_profiles.py --runs "baselines/**/init.pt" \
        --data-root data --out profiles/baseline --baseline

Two seams need to match your codebase; both are marked ADAPTER below.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

import attention_common as C
from build_predictors import CONDITION_REGEX, find_conditions


# --------------------------------------------------------------------------
# ADAPTER 1: turning a checkpoint path into a model
# --------------------------------------------------------------------------

def load_model(ckpt_path: str, device: str, factory: str | None):
    """Return an eval-mode nn.Module.

    If --model-factory is given as "module:function", that function is called
    with the loaded checkpoint dict and must return the model. Otherwise a
    checkpoint holding a pickled module (directly or under 'model') is used.
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if factory:
        mod_name, fn_name = factory.split(":")
        sys.path.insert(0, os.getcwd())
        fn = getattr(__import__(mod_name, fromlist=[fn_name]), fn_name)
        model = fn(ckpt)
    elif isinstance(ckpt, torch.nn.Module):
        model = ckpt
    elif isinstance(ckpt, dict) and isinstance(ckpt.get("model"), torch.nn.Module):
        model = ckpt["model"]
    else:
        raise RuntimeError(
            f"{ckpt_path} holds a state_dict, not a model. Pass "
            "--model-factory module:function pointing at a builder that takes "
            "the checkpoint dict and returns the constructed model."
        )
    return model.to(device).eval(), ckpt


# --------------------------------------------------------------------------
# ADAPTER 2: held-out inputs for a condition
# --------------------------------------------------------------------------

def load_inputs(data_dir: str, n_inputs: int, seed: int = 0) -> np.ndarray:
    """Return (n_inputs, N) held-out snapshots u^n.

    Looks for test.npy / test_u.npy / trajectories.npy under data_dir and
    flattens all leading axes into a snapshot axis. Adjust the candidate list
    if your generator writes different filenames.
    """
    candidates = ["test.npy", "test_u.npy", "u_test.npy", "trajectories.npy", "u.npy"]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            arr = np.load(path)
            break
    else:
        raise FileNotFoundError(
            f"no input array in {data_dir}; looked for {candidates}"
        )
    arr = np.asarray(arr, dtype=np.float32).reshape(-1, arr.shape[-1])
    rng = np.random.default_rng(seed)
    if arr.shape[0] > n_inputs:
        arr = arr[rng.choice(arr.shape[0], n_inputs, replace=False)]
    return arr


def to_batch(x: np.ndarray, device: str):
    """Shape held-out snapshots into whatever the model's forward expects.

    Default assumes (B, N) with an internal embedding. If your forward takes
    (B, N, 1), add the trailing axis here.
    """
    import torch

    return torch.from_numpy(x).to(device)


# --------------------------------------------------------------------------

_DATA_FILE_NAMES = ("test.npy", "test_u.npy", "u_test.npy", "trajectories.npy",
                    "u.npy", "train.npy")


def _has_data(d: str) -> bool:
    """True if d directly contains a trajectory array load_inputs()/
    load_eval_pairs() would accept. Checking this instead of just
    os.path.isdir() matters: a condition directory that holds only OTHER
    seeds' subdirectories (e.g. data/lax_friedrichs_r_0.25 with seed_305..309
    on disk but not this checkpoint's seed_300) is a real, existing directory
    that nonetheless has no data at its own top level, and treating it as a
    hit sent every fallback straight past the seed-mismatch handling below.
    """
    return any(os.path.exists(os.path.join(d, name)) for name in _DATA_FILE_NAMES)


def _seed_or_flat(cond_dir: str, seed: int) -> str | None:
    """Prefer cond_dir/seed_<seed> (the real per-seed layout this project's
    generators write: run_heat.py/run_friedrichs.py point --save_dir at
    "<condition>/seed_<N>", so the condition directory itself usually has no
    files, only seed_* subdirectories). Falls back to cond_dir itself, since
    one legacy directory (heat_0.25) has data at that flat level too.
    Returns None if neither actually holds data, so the caller can keep
    trying rather than being handed an empty or wrong-seed directory.
    """
    seeded = os.path.join(cond_dir, f"seed_{seed}")
    if _has_data(seeded):
        return seeded
    if _has_data(cond_dir):
        return cond_dir
    return None


def condition_data_dir(data_root: str, meta: C.RunMeta, template: str,
                        condition_regex: str) -> str:
    """Locate the data directory for one condition, matched to one seed.

    Tries three things in order, preferring cond_dir/seed_<meta.seed> over
    cond_dir itself at each step (see _seed_or_flat):
      1. The checkpoint's own condition-directory basename, reused verbatim
         under data_root. This project's run scripts (run_heat.py,
         run_friedrichs.py, batch_metrics.py) all derive CKPT_ROOT and
         DATA_ROOT from the same tag, so this is the precise match -- and it
         matters, not just as a nicety: more than one data directory can
         parse to the same (pde, r) (both "heat_0.25" and "heat_new_0.25"
         exist on disk, and heat_0.25's stencil.npy on disk does not even
         match its own name -- it holds r=0.1024) and only one of them was
         actually used to train this checkpoint. Picking the wrong one
         wouldn't error, it would just silently feed the model held-out
         snapshots that don't match what it trained on.
      2. The literal --data-template.
      3. A regex scan of data_root for any directory parsing to the same
         (pde, r) via CONDITION_REGEX -- the same parsing already applied to
         checkpoint paths, so both sides agree on what a condition is. Warns
         if more than one such directory exists, since the choice is then
         genuinely ambiguous and this fallback has no way to break the tie
         correctly.
    """
    ckpt_cond_dir = os.path.basename(os.path.dirname(os.path.dirname(meta.path)))
    cond_dirs = [os.path.join(data_root, ckpt_cond_dir),
                 os.path.join(data_root, template.format(pde=meta.pde, r=f"{meta.r:g}"))]

    for cd in cond_dirs:
        hit = _seed_or_flat(cd, meta.seed)
        if hit:
            return hit

    candidates = [path for pde, r, path in find_conditions(data_root, condition_regex)
                  if pde == meta.pde and abs(r - meta.r) < 1e-9]
    if len(candidates) > 1:
        print(f"warning: {len(candidates)} data directories match {meta.pde} "
              f"r={meta.r:g} ({', '.join(candidates)}); trying each for "
              f"seed_{meta.seed} or a flat layout, in discovery order.",
              file=sys.stderr)
    cond_dirs += candidates
    for c in candidates:
        hit = _seed_or_flat(c, meta.seed)
        if hit:
            return hit

    # No directory has this checkpoint's own seed. This project's training
    # scripts delete a seed's generated data after that seed finishes
    # training (to save disk; see compute_random_baselines.py's
    # --regenerate_data), so checkpoint seeds and remaining data seeds
    # routinely don't overlap -- confirmed on real data, where e.g.
    # lax_friedrichs_r_0.1 has checkpoints for seeds 300-319 but data only
    # for seed 300. Falling back to any other available seed of the same
    # (pde, r) is still statistically valid for evaluation -- every seed of
    # a condition is an independent draw from the same generator -- but it
    # is a real approximation (this checkpoint's *own* held-out split may no
    # longer exist), so it is logged rather than applied silently.
    for cd in cond_dirs:
        seed_dirs = [d for d in sorted(glob.glob(os.path.join(cd, "seed_*")))
                    if _has_data(d)]
        if seed_dirs:
            print(f"warning: no seed_{meta.seed} data for {meta.pde} "
                  f"r={meta.r:g}; evaluating against {seed_dirs[0]} instead "
                  "(a different seed of the same condition, since this "
                  "project's training scripts delete data after use).",
                  file=sys.stderr)
            return seed_dirs[0]

    # nothing matched at all; return the most informative guess so
    # load_inputs()'s FileNotFoundError points somewhere sensible
    return os.path.join(data_root, ckpt_cond_dir, f"seed_{meta.seed}")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)  # prefix matching bit us before
    ap.add_argument("--runs", required=True,
                    help="glob for checkpoints, e.g. 'runs/**/best.pt'")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--data-template", default="{pde}_r{r}",
                    help="subdirectory name per condition; tried first, before "
                         "the --condition-regex fallback scan")
    ap.add_argument("--condition-regex", default=CONDITION_REGEX,
                    help="fallback for locating a condition's data directory "
                         "when --data-template doesn't match anything on disk")
    ap.add_argument("--out", default="profiles/trained")
    ap.add_argument("--n-inputs", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-factory", default=None,
                    help="module:function returning a model from a checkpoint dict")
    ap.add_argument("--attention-extractor", default=None,
                    help="module:function(model, batch) -> (n_layers, n_heads, N, N), "
                         "overriding attention_common.extract_attention. Needed whenever the "
                         "model's own attention module takes an internal fast path "
                         "that bypasses attention_common's monkeypatches -- confirmed true "
                         "for this project's VanillaTransformer; pass "
                         "model_adapter:extract_attention.")
    ap.add_argument("--run-regex", default=C.DEFAULT_RUN_REGEX)
    ap.add_argument("--baseline", action="store_true",
                    help="tag these profiles as random-init baselines")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print parsed (pde, r, seed) per checkpoint and exit")
    args = ap.parse_args()

    if args.attention_extractor:
        mod_name, fn_name = args.attention_extractor.split(":")
        sys.path.insert(0, os.getcwd())
        extract_attention = getattr(__import__(mod_name, fromlist=[fn_name]), fn_name)
    else:
        extract_attention = C.extract_attention

    paths = sorted(glob.glob(args.runs, recursive=True))
    if not paths:
        sys.exit(f"no checkpoints matched {args.runs}")

    parsed, unparsed = [], []
    for p in paths:
        meta = C.parse_run(p, args.run_regex)
        (parsed if meta else unparsed).append(meta or p)

    if args.dry_run:
        for m in parsed:
            print(f"{m.pde:5s} r={m.r:<6g} seed={m.seed:<5d} {m.path}")
        for p in unparsed:
            print(f"UNPARSED  {p}")
        print(f"\n{len(parsed)} parsed, {len(unparsed)} unparsed")
        return
    if unparsed:
        print(f"warning: {len(unparsed)} checkpoints did not match --run-regex; "
              "rerun with --dry-run to inspect", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    input_cache: dict[str, np.ndarray] = {}

    for i, meta in enumerate(parsed, 1):
        tag = f"{meta.pde}_r{meta.r:g}_seed{meta.seed}"
        out_path = os.path.join(args.out, tag + ".npz")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{i}/{len(parsed)}] skip {tag} (exists)")
            continue

        ddir = condition_data_dir(args.data_root, meta, args.data_template,
                                   args.condition_regex)
        if ddir not in input_cache:
            input_cache[ddir] = load_inputs(ddir, args.n_inputs)
        x = input_cache[ddir]

        model, ckpt = load_model(meta.path, args.device, args.model_factory)
        attn = extract_attention(model, to_batch(x, args.device))
        n = attn.shape[-1]
        offsets = C.offsets_full(n)
        profile = C.relative_profile(attn, offsets)  # (L, H, len(offsets))

        row_sums = attn.sum(axis=-1)
        np.savez_compressed(
            out_path,
            profile=profile.astype(np.float32),
            offsets=offsets,
            pde=meta.pde,
            r=meta.r,
            seed=meta.seed,
            n_layers=attn.shape[0],
            n_heads=attn.shape[1],
            n_space=n,
            n_inputs=x.shape[0],
            is_baseline=bool(args.baseline),
            ckpt=meta.path,
            row_sum_min=float(row_sums.min()),
            row_sum_max=float(row_sums.max()),
        )
        print(f"[{i}/{len(parsed)}] {tag}: L={attn.shape[0]} H={attn.shape[1]} "
              f"N={n} rowsum=[{row_sums.min():.4f}, {row_sums.max():.4f}]")

        del model, attn, ckpt
        if args.device.startswith("cuda"):
            import torch
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
