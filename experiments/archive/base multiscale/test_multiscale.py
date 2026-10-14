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

from src.models.multiscale_pinn import (
    MultiScalePINN,
)


# ============================================================
# Main Test
# ============================================================

def main():

    print()
    print("=" * 65)
    print("Three-Scale PINN Architecture Test")
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
    # Test input
    # --------------------------------------------------------

    x = torch.linspace(
        0.0,
        1.0,
        5,
    ).reshape(-1, 1)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    output = model(x)

    # --------------------------------------------------------
    # Number of parameters
    # --------------------------------------------------------

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("Input shape:")
    print(x.shape)

    print()
    print("Output shape:")
    print(output.shape)

    print()
    print("Model output:")
    print(output)

    print()
    print("Number of parameters:")
    print(number_of_parameters)

    print()
    print("=" * 65)
    print("Architecture test completed successfully.")
    print("=" * 65)
    print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()  