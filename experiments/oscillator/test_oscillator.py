import os
import sys

import torch


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.models.pinn import PINN
from src.equations.ode_oscillator import residual
from src.equations.ode_oscillator import exact_solution


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(42)


# ============================================================
# Model
# ============================================================

model = PINN(
    input_dim=1,
    hidden_dim=32,
    hidden_layers=3,
    output_dim=1,
)


# ============================================================
# Test Points
# ============================================================

x = torch.linspace(
    0.0,
    2.0,
    5,
).reshape(-1, 1)


# ============================================================
# Exact Solution
# ============================================================

exact = exact_solution(x)


# ============================================================
# Physics Residual
# ============================================================

r = residual(
    model,
    x,
)


# ============================================================
# Output
# ============================================================

print()
print("=" * 65)
print("Harmonic Oscillator - Second Derivative Test")
print("=" * 65)

print()

print("Input x:")
print(x)

print()

print("Exact solution sin(x):")
print(exact)

print()

print("Physics residual r(x) = u'' + u:")
print(r)

print()

print("Shapes:")
print(
    "x       :",
    x.shape,
)

print(
    "exact   :",
    exact.shape,
)

print(
    "residual:",
    r.shape,
)

print()

print("=" * 65)