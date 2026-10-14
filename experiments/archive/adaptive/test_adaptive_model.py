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
# Import
# ============================================================

from src.models.adaptive_multiscale_pinn import (
    AdaptiveMultiScalePINN
)


# ============================================================
# Main
# ============================================================

def main():

    torch.manual_seed(42)

    print("=" * 70)
    print("Adaptive Three-Scale PINN Architecture Test")
    print("=" * 70)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AdaptiveMultiScalePINN(
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
    # Test Input
    # --------------------------------------------------------

    x = torch.linspace(
        0.0,
        1.0,
        5,
    ).reshape(-1, 1)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    output = model(x)

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    # --------------------------------------------------------
    # Scale Weights
    # --------------------------------------------------------

    weights = model.get_scale_weights()

    # --------------------------------------------------------
    # Results
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
    print("Adaptive scale weights:")

    print(
        f"Low  Scale Weight : "
        f"{weights[0].item():.6f}"
    )

    print(
        f"Mid  Scale Weight : "
        f"{weights[1].item():.6f}"
    )

    print(
        f"High Scale Weight : "
        f"{weights[2].item():.6f}"
    )

    print()
    print(
        f"Weight Sum: "
        f"{weights.sum().item():.6f}"
    )

    print()
    print("=" * 70)
    print(
        "Adaptive architecture test "
        "completed successfully."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()