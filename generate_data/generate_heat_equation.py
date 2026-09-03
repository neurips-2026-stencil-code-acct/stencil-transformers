"""
generate_heat_equation_data.py

Generates synthetic training data for the 1D heat equation using explicit
finite difference simulation. Creates train/val/test trajectory datasets
for transformer models to learn PDE dynamics.

Physics:
    - Solves: du/dt = alpha * d²u/dx² (1D diffusion equation)
    - Uses explicit FTCS (Forward-Time Central-Space) finite difference scheme
    - Periodic boundary conditions with CFL stability constraint

Data Format:
    - Each trajectory: (NT+1, NX) array of spatial snapshots over time
    - Random Fourier mode initial conditions for diverse training data
    - Outputs: train.npy, val.npy, test.npy, stencil.npy

Usage:
    python generate_heat_equation_data.py --alpha 0.25 --seed 42 --save_dir data/

References:
    - Takamoto et al., NeurIPS 2022 (PDEBench)
    - LeVeque, Finite Difference Methods for ODEs and PDEs, 2007
"""

import numpy as np
import os
import argparse

# ── Grid parameters ────────────────────────────────────────────────────────────
NX = 64  # spatial grid points (Takamoto et al., NeurIPS 2022)
NT = 2000 # timesteps per trajectory justify later 
DX = 1.0 / NX # spatial resolution
DT = 0.0001 # timestep (must satisfy CFL: alpha*dt/dx^2 <= 0.5) Design choice in section 3.1 
# ALPHA = 0.25 # thermal diffusivity, material property that controls how fast heat spreads through the rod 

N_TRAIN = 800
N_VAL = 100
N_TEST = 100
# SEED = 42 # Replicability 

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.25)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_dir", type=str, default=None)
args = parser.parse_args()

r = args.alpha
SEED = args.seed
# CFL check

'''
Numerical Stability Constant (Courant, 1967)
'''
#r = ALPHA * DT / DX**2
assert r <= 0.5, f"CFL violated: r={r:.4f}. Reduce DT or increase DX."
print(f"CFL number r = {r:.4f}  (must be <= 0.5)")

# Stencil coefficients (2nd order derivative, central difference for heat equation) ──────
# u_t = alpha * (u_{i-1} - 2*u_i + u_{i+1}) / dx^2
# Discrete update: u_new[i] = u[i] + r*(u[i-1] - 2*u[i] + u[i+1])
# Stencil weights over [-1, 0, +1] neighborhood:

'''
Spatial Second derivative using second order central difference 
Cite Leveque 2007

'''
STENCIL = np.array([r, 1 - 2*r, r])  

    # [left, center, right]
print(f"FD stencil coefficients: {STENCIL}")
print(f"  left={STENCIL[0]:.4f}, center={STENCIL[1]:.4f}, right={STENCIL[2]:.4f}")

# ── Initial condition generator ────────────────────────────────────────────────
'''
Cite Takamoto et al 2022 
'''
def random_ic(nx, rng, n_modes=5):
    """Sum of random Fourier modes — smooth, periodic initial conditions."""
    x = np.linspace(0, 1, nx, endpoint=False)
    u = np.zeros(nx)
    for k in range(1, n_modes + 1):
        amp = rng.uniform(-1, 1)
        phase = rng.uniform(0, 2 * np.pi)
        u += amp * np.sin(2 * np.pi * k * x + phase)
    return u / (np.abs(u).max() + 1e-8)   # normalize to [-1, 1]

# ── Single trajectory simulator ────────────────────────────────────────────────
'''
Leveque, Takamoto 
'''
def simulate(u0, nt, r):
    """
    Simulate heat equation via explicit finite difference.
    Periodic boundary conditions.
    Returns array of shape (nt+1, nx).
    """
    nx = len(u0)
    traj = np.zeros((nt + 1, nx), dtype=np.float32)
    traj[0] = u0
    u = u0.copy()
    for t in range(nt):
        # np.roll handles periodic BCs cleanly
        u_left = np.roll(u,  1)
        u_right = np.roll(u, -1)
        u = u + r * (u_left - 2*u + u_right)
        traj[t + 1] = u
    return traj

# ── Dataset generation ─────────────────────────────────────────────────────────
'''

'''
def make_dataset(n, rng):
    """Generate n trajectories. Returns (n, nt+1, nx) array"""
    trajs = []
    for _ in range(n):
        u0 = random_ic(NX, rng)
        traj = simulate(u0, NT, r)
        trajs.append(traj)
    return np.stack(trajs)   # (n, nt+1, nx)

rng = np.random.default_rng(SEED)
print("\nGenerating trajectories...")
train = make_dataset(N_TRAIN, rng)
val = make_dataset(N_VAL, rng)
test = make_dataset(N_TEST, rng)

print(f"  Train: {train.shape}  Val: {val.shape}  Test: {test.shape}")

# ── Save ───────────────────────────────────────────────────────────────────────
#save_dir = f"data/alpha_{ALPHA}"
if args.save_dir:
    save_dir = args.save_dir
else:
    save_dir = f"data/alpha_{ALPHA}"

os.makedirs(save_dir, exist_ok=True)

os.makedirs(save_dir, exist_ok=True)
np.save(f"{save_dir}/train.npy", train)
np.save(f"{save_dir}/val.npy",   val)
np.save(f"{save_dir}/test.npy",  test)
np.save(f"{save_dir}/stencil.npy", STENCIL)

print(f"\nSaved to {save_dir}/")
print("train.npy, val.npy, test.npy, stencil.npy")
print(f"\nEach sample: {NT} input steps --> predict next step")
print("Ready for tokenization.")