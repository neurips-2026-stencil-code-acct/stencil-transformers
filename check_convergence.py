"""
check_convergence_single.py (windowed-mean version)

Check whether a single seed's val_losses.npy has converged, using an
absolute-improvement threshold comparing the MEAN of the first half of
the trailing window against the MEAN of the second half — this averages
out single-epoch noise (e.g. a random spike right at the window boundary)
that a raw endpoint-to-endpoint comparison is vulnerable to.
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str,
                     default="checkpoints/heat_0.1/seed_100/val_losses.npy",
                     help="Path to val_losses.npy")
parser.add_argument("--window", type=int, default=8,
                     help="Trailing window size (epochs) to measure improvement over. "
                          "Must be even to split cleanly in half.")
parser.add_argument("--abs_threshold", type=float, default=1e-9,
                     help="Absolute improvement threshold below which we call it converged")
args = parser.parse_args()

if args.window % 2 != 0:
    raise ValueError(f"--window must be even to split into two halves (got {args.window})")

val_losses = np.load(args.path)
n_epochs = len(val_losses)
half = args.window // 2

print(f"File: {args.path}")
print(f"Epochs recorded: {n_epochs}")
print(f"Val losses: {np.array2string(val_losses, precision=4, suppress_small=False)}")

if n_epochs <= args.window:
    print(f"\nToo short to judge (n_epochs <= window={args.window})")
    raise SystemExit

trailing = val_losses[-args.window:]
first_half = trailing[:half]
second_half = trailing[half:]

first_mean = first_half.mean()
second_mean = second_half.mean()
best_loss = val_losses.min()
final_loss = val_losses[-1]

abs_improve = first_mean - second_mean
rel_improve = abs_improve / first_mean if first_mean != 0 else float("nan")
converged = abs_improve < args.abs_threshold
best_at_final = np.isclose(best_loss, final_loss)

print(f"\nBest loss (min over all epochs): {best_loss:.4e}")
print(f"Final loss (last epoch):         {final_loss:.4e}")
print(f"Window: last {args.window} epochs "
      f"(epoch {n_epochs - args.window} -> {n_epochs})")
print(f"First-half mean  (epochs {n_epochs - args.window}-{n_epochs - half - 1}): {first_mean:.4e}")
print(f"Second-half mean (epochs {n_epochs - half}-{n_epochs - 1}):  {second_mean:.4e}")
print(f"Absolute improvement (first-half mean -> second-half mean): {abs_improve:.4e}")
print(f"Relative improvement: {rel_improve*100:.2f}%")
print(f"Absolute threshold: {args.abs_threshold:.1e}")

status = "CONVERGED" if converged else "STILL_IMPROVING"
suffix = " (best loss at final epoch)" if best_at_final and not converged else ""
print(f"\nStatus: {status}{suffix}")