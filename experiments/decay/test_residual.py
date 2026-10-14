import os
import sys

import torch

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.pinn import PINN # type: ignore
from src.equations.ode_decay import residual # type: ignore


def main():

    # --------------------------------------------------
    # 1. Create model
    # --------------------------------------------------

    model = PINN(
        input_dim=1,
        hidden_dim=32,
        hidden_layers=3,
        output_dim=1,
    )

    # --------------------------------------------------
    # 2. Collocation points
    # --------------------------------------------------

    x = torch.linspace(
        0.0,
        1.0,
        5,
    ).reshape(-1, 1)

    # --------------------------------------------------
    # 3. Compute residual
    # --------------------------------------------------

    residual_value = residual(
        model,
        x,
    )

    # --------------------------------------------------
    # 4. Compute physics loss
    # --------------------------------------------------

    physics_loss = torch.mean(
        residual_value ** 2
    )

    # --------------------------------------------------
    # 5. Print results
    # --------------------------------------------------

    print("=" * 60)
    print("PINN Residual Test")
    print("=" * 60)

    print("\nCollocation points:")
    print(x)

    print("\nResidual r(x) = du/dx + u:")
    print(residual_value)

    print("\nPhysics Loss:")
    print(physics_loss.item())

    print("\nResidual shape:")
    print(residual_value.shape)

    print("=" * 60)


if __name__ == "__main__":
    main()