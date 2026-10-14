import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Basic Physics-Informed Neural Network.

    The network maps the input coordinate(s) to
    the predicted solution u.
    """

    def __init__(
        self,
        input_dim=1,
        hidden_dim=32,
        hidden_layers=3,
        output_dim=1,
    ):
        super().__init__()

        layers = []

        # Input layer
        layers.append(
            nn.Linear(input_dim, hidden_dim)
        )
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(
                nn.Linear(hidden_dim, hidden_dim)
            )
            layers.append(nn.Tanh())

        # Output layer
        layers.append(
            nn.Linear(hidden_dim, output_dim)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)