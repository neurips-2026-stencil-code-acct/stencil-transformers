"""Convert saved random-attention matrices into baseline profile files.

``compute_random_baselines.py`` saves matrices of shape
``(random_seed, layer, head, query, key)``. Attention assignment instead consumes
one relative-offset profile file per initialization, with condition metadata.
This converter bridges those formats without rerunning untrained models.

Usage
-----
python convert_random_baselines.py \\
    --baseline-root ../../final_results/random_baselines \\
    --out profiles/baseline --overwrite
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

import attention_common as C
from build_predictors import CONDITION_REGEX


def parse_condition(name: str) -> tuple[str, float] | None:
    match = re.search(CONDITION_REGEX, name, flags=re.IGNORECASE)
    if not match:
        return None
    pde = C.PDE_ALIASES.get(match.group("pde").lower(), match.group("pde").lower())
    return pde, float(match.group("r"))


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--baseline-root", default="../../final_results/random_baselines")
    parser.add_argument("--out", default="profiles/baseline")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="fail if any condition directory lacks saved attention matrices")
    args = parser.parse_args()

    root = os.path.abspath(args.baseline_root)
    if not os.path.isdir(root):
        parser.error(f"baseline root does not exist: {root}")
    os.makedirs(args.out, exist_ok=True)

    written = 0
    skipped = []
    for name in sorted(os.listdir(root)):
        cond_dir = os.path.join(root, name)
        if not os.path.isdir(cond_dir):
            continue
        parsed = parse_condition(name)
        if parsed is None:
            continue
        pde, r = parsed
        source = os.path.join(cond_dir, "random_baseline_mean_attn.npy")
        if not os.path.exists(source):
            skipped.append(name)
            continue
        matrices = np.load(source, mmap_mode="r")
        if matrices.ndim != 5 or matrices.shape[-1] != matrices.shape[-2]:
            raise ValueError(
                f"{source}: expected (seed, layer, head, N, N), got {matrices.shape}"
            )
        n_random, n_layers, n_heads, n_space, _ = matrices.shape
        offsets = C.offsets_full(n_space)
        for seed in range(n_random):
            destination = os.path.join(args.out, f"{pde}_r{r:g}_seed{seed}.npz")
            if os.path.exists(destination) and not args.overwrite:
                continue
            profile = C.relative_profile(np.asarray(matrices[seed]), offsets)
            row_sum = matrices[seed].sum(axis=-1)
            np.savez_compressed(
                destination,
                profile=profile.astype(np.float32),
                offsets=offsets,
                pde=pde,
                r=r,
                seed=seed,
                n_layers=n_layers,
                n_heads=n_heads,
                n_space=n_space,
                n_inputs=-1,
                is_baseline=True,
                ckpt=f"random_initialization:{source}:seed_{seed}",
                row_sum_min=float(row_sum.min()),
                row_sum_max=float(row_sum.max()),
            )
            written += 1
        print(f"{name}: converted {n_random} random initializations ({n_layers}L/{n_heads}H)")

    if skipped:
        message = "missing random_baseline_mean_attn.npy for: " + ", ".join(skipped)
        if args.strict:
            parser.error(message)
        print(f"warning: {message}", file=sys.stderr)
    if not written and not skipped:
        parser.error(f"no baseline condition directories were found under {root}")
    print(f"wrote {written} baseline attention files to {args.out}")


if __name__ == "__main__":
    main()
