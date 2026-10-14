import torch

from src.models.AMS_PINN import (
    AdaptiveMultiScalePINN,
)


def main():

    print("=" * 70)
    print(
        "Adaptive Multi-Scale Fourier PINN Test"
    )
    print("=" * 70)

    torch.manual_seed(42)

    model = AdaptiveMultiScalePINN(
        input_dim=1,
        hidden_dim=16,
        feature_dim=16,
        fusion_dim=32,
        output_dim=1,
        hidden_layers=2,
        initial_low_frequency=1.0,
        initial_mid_frequency=3.0,
        initial_high_frequency=10.0,
        trainable_frequencies=True,
    )

    # --------------------------------------------------------
    # Forward test
    # --------------------------------------------------------

    x = torch.linspace(
        0.0,
        1.0,
        5,
    ).reshape(-1, 1)

    x.requires_grad_(True)

    u = model(x)

    print()
    print("Input shape:")
    print(x.shape)

    print()
    print("Output shape:")
    print(u.shape)

    print()
    print("Model output:")
    print(u)

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print("Number of parameters:")
    print(parameter_count)

    # --------------------------------------------------------
    # Frequencies
    # --------------------------------------------------------

    frequencies = (
        model.get_frequencies()
        .detach()
    )

    print()
    print("Adaptive Frequencies:")
    print(
        f"Low  Frequency : "
        f"{frequencies[0].item():.6f}"
    )

    print(
        f"Mid  Frequency : "
        f"{frequencies[1].item():.6f}"
    )

    print(
        f"High Frequency : "
        f"{frequencies[2].item():.6f}"
    )

    # --------------------------------------------------------
    # Scale weights
    # --------------------------------------------------------

    weights = (
        model.get_scale_weights()
        .detach()
    )

    print()
    print("Adaptive Scale Weights:")

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
        "Weight Sum:"
    )
    print(
        f"{weights.sum().item():.6f}"
    )

    # --------------------------------------------------------
    # Derivative test
    # --------------------------------------------------------

    du_dx = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    d2u_dx2 = torch.autograd.grad(
        outputs=du_dx,
        inputs=x,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True,
    )[0]

    print()
    print("First derivative du/dx:")
    print(du_dx)

    print()
    print("Second derivative d2u/dx2:")
    print(d2u_dx2)

    print()
    print("Derivative shapes:")
    print("du/dx   :", du_dx.shape)
    print("d2u/dx2 :", d2u_dx2.shape)

    print()
    print("=" * 70)
    print(
        "Adaptive Multi-Scale PINN test "
        "completed successfully."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()