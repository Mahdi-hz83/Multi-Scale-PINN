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


# ============================================================
# Import Model
# ============================================================

from src.models.multiscale_pinn import MultiScalePINN


# ============================================================
# Derivative Function
# ============================================================

def compute_derivative(
    y,
    x,
):
    """
    Compute dy/dx using PyTorch automatic differentiation.
    """

    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


# ============================================================
# Main Test
# ============================================================

def main():

    print()
    print("=" * 65)
    print("Three-Scale PINN - Automatic Differentiation Test")
    print("=" * 65)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = MultiScalePINN(
        input_dim=1,
        hidden_dim=16,
        feature_dim=16,
        fusion_hidden_dim=16,
        output_dim=1,
        hidden_layers=2,
        low_frequency=1.0,
        mid_frequency=3.0,
        high_frequency=10.0,
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    x = torch.linspace(
        0.0,
        1.0,
        5,
    ).reshape(-1, 1)

    x.requires_grad_(True)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    u = model(x)

    # --------------------------------------------------------
    # First derivative
    # --------------------------------------------------------

    du_dx = compute_derivative(
        u,
        x,
    )

    # --------------------------------------------------------
    # Second derivative
    # --------------------------------------------------------

    d2u_dx2 = compute_derivative(
        du_dx,
        x,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("Input x:")
    print(x)

    print()
    print("Network output u(x):")
    print(u)

    print()
    print("First derivative du/dx:")
    print(du_dx)

    print()
    print("Second derivative d2u/dx2:")
    print(d2u_dx2)

    print()
    print("Shapes:")
    print("x       :", x.shape)
    print("u       :", u.shape)
    print("du/dx   :", du_dx.shape)
    print("d2u/dx2 :", d2u_dx2.shape)

    print()
    print("=" * 65)
    print("Automatic differentiation test completed successfully.")
    print("=" * 65)
    print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()