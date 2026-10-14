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


def main():

    # Create model
    model = PINN(
        input_dim=1,
        hidden_dim=32,
        hidden_layers=3,
        output_dim=1,
    )

    # Test input
    x = torch.tensor(
        [[0.0],
         [0.25],
         [0.5],
         [0.75],
         [1.0]],
        dtype=torch.float32,
    )

    # Forward pass
    output = model(x)

    print("=" * 50)
    print("PINN Model Test")
    print("=" * 50)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    print("\nModel output:")
    print(output)

    print("\nNumber of parameters:")
    print(
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    print("=" * 50)


if __name__ == "__main__":
    main()