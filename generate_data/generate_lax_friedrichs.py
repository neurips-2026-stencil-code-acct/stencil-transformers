"""
generate_advection_lax_friedrichs_data.py

Generates synthetic training data for the 1D linear advection equation using
the Lax-Friedrichs explicit finite difference scheme. Creates train/val/test
trajectory datasets for transformer models to learn PDE dynamics.

Physics:
    - Solves: du/dt + c * du/dx = 0 (1D linear advection equation)
    - Uses explicit Lax-Friedrichs finite difference scheme (1st-order
      accurate, strongly numerically diffusive)
    - Periodic boundary conditions with CFL stability constraint (|r| <= 1)

Why Lax-Friedrichs (interpretability rationale):
    Unlike every other scheme in this project (heat FTCS, wave leapfrog,
    advection Lax-Wendroff), Lax-Friedrichs replaces the center point u[i]
    entirely with the average of its two neighbors -- the true update rule
    assigns ZERO weight to the token's own previous value. Same radius-1,
    2-time-level structure as Lax-Wendroff, but with a hole at the center.
    This isolates whether learned attention reproduces the literal operator
    (should show zero weight on the diagonal) or generic local correlation
    (should still show a self-weight, since neighboring smooth values are
    highly correlated with self regardless of what the true stencil says).

Stencil:
    u_new[i] = (1+r)/2 * u[i-1]  +  0 * u[i]  +  (1-r)/2 * u[i+1]
    All-positive, zero center weight, radius 1, 2 time levels (single-step).
    Frequency response is complex (dispersive) with strong numerical damping
    (|G(k)| well below Lax-Wendroff's across the sweep range) -- see
    companion note on von Neumann analysis across heat/wave/advection.

Data Format:
    - Each trajectory: (NT+1, NX) array of spatial snapshots over time
    - Random Fourier mode initial conditions for diverse training data
    - Outputs: train.npy, val.npy, test.npy, stencil.npy

Usage:
    python generate_advection_lax_friedrichs_data.py --r 0.25 --seed 42 --save_dir data_advection_lf/

References:
    - LeVeque, Finite Difference Methods for ODEs and PDEs, 2007
    - Courant, Friedrichs, Lewy, 1967 (CFL stability condition)
"""

import numpy as np
import os
import argparse

# ── Grid parameters ────────────────────────────────────────────────────────────
# Kept identical to heat/wave/Lax-Wendroff for cross-scheme comparability.
NX = 64
NT = 2000
DX = 1.0 / NX
DT = 0.0001

N_TRAIN = 800
N_VAL = 100
N_TEST = 100

parser = argparse.ArgumentParser()
parser.add_argument("--r", type=float, default=0.25)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_dir", type=str, default="data_advection_lf")
args = parser.parse_args()

R = args.r
SEED = args.seed

# Derive advection speed c from r: r = c * dt / dx  (Courant number)
C = R * DX / DT
print(f"r = {R:.4f}, c = {C:.4f}")

# CFL check (Courant et al., 1967)
assert abs(R) <= 1.0, f"CFL violated: r={R:.4f}. Must satisfy |r| <= 1.0."
print(f"CFL number r = {R:.4f}  (must be <= 1.0, Lax-Friedrichs)")

# Stencil: u_new[i] = (1+r)/2 * u[i-1]  +  0 * u[i]  +  (1-r)/2 * u[i+1]
# The center point is replaced by the average of its neighbors (the
# stabilizing modification that distinguishes Lax-Friedrichs from a naive,
# unconditionally unstable central-difference advection scheme).

STENCIL = np.array([(1 + R) / 2, 0.0, (1 - R) / 2])
print(f"FD stencil coefficients: {STENCIL}")
print(f"  left={STENCIL[0]:.4f}, center={STENCIL[1]:.4f}, right={STENCIL[2]:.4f}")

def random_ic(nx, rng, n_modes=5):
    """Sum of random Fourier modes — smooth, periodic initial conditions."""
    x = np.linspace(0, 1, nx, endpoint=False)
    u = np.zeros(nx)
    for k in range(1, n_modes + 1):
        amp = rng.uniform(-1, 1)
        phase = rng.uniform(0, 2 * np.pi)
        u += amp * np.sin(2 * np.pi * k * x + phase)
    return u / (np.abs(u).max() + 1e-8)

def simulate(u0, nt, r):
    """
    Simulate advection equation via Lax-Friedrichs finite difference.
    Periodic boundary conditions. Single-step (2-level) method, no bootstrap.
    Returns array of shape (nt+1, nx).
    """
    nx = len(u0)
    traj = np.zeros((nt + 1, nx), dtype=np.float32)
    traj[0] = u0
    u = u0.copy()
    for t in range(nt):
        u_left  = np.roll(u,  1)
        u_right = np.roll(u, -1)
        u = 0.5 * (u_left + u_right) - (r / 2) * (u_right - u_left)
        traj[t + 1] = u
    return traj

def make_dataset(n, rng):
    trajs = []
    for _ in range(n):
        u0 = random_ic(NX, rng)
        traj = simulate(u0, NT, R)
        trajs.append(traj)
    return np.stack(trajs)

rng = np.random.default_rng(SEED)
print("\nGenerating trajectories...")
train = make_dataset(N_TRAIN, rng)
val   = make_dataset(N_VAL,   rng)
test  = make_dataset(N_TEST,  rng)
print(f"  Train: {train.shape}  Val: {val.shape}  Test: {test.shape}")

if args.save_dir:
    save_dir = args.save_dir
else:
    save_dir = f"data/advection_lf_r_{R}"

os.makedirs(save_dir, exist_ok=True)
np.save(f"{save_dir}/train.npy",   train)
np.save(f"{save_dir}/val.npy",     val)
np.save(f"{save_dir}/test.npy",    test)
np.save(f"{save_dir}/stencil.npy", STENCIL)

print(f"\nSaved to {save_dir}/")
print(f"r={R:.4f}, stencil={STENCIL}")
print(f"Each sample: {NT} input steps --> predict next step")