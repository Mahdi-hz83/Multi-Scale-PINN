"""Adaptive Multi-Scale PINN without Fourier features.

Each branch receives the same raw coordinates.  Multi-scale behaviour is
created by using different branch capacities/depths and learning adaptive
branch weights before residual feature fusion.  Tanh is used throughout so
first- and higher-order automatic differentiation remains smooth for PINNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(max(0, hidden_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ResidualFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        shortcut = self.projection(x)
        return self.activation(shortcut + self.block(shortcut))


class AdaptiveMultiScalePINN(nn.Module):
    """Three-branch adaptive Multi-Scale PINN, without Fourier encoding."""

    def __init__(
        self,
        input_dim=1,
        hidden_dim=16,
        feature_dim=16,
        fusion_dim=32,
        output_dim=1,
        hidden_layers=2,
        branch_hidden_dims=None,
        branch_hidden_layers=None,
        initial_low_frequency=None,
        initial_mid_frequency=None,
        initial_high_frequency=None,
        trainable_frequencies=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.fusion_dim = fusion_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.trainable_frequencies = False

        if branch_hidden_dims is None:
            branch_hidden_dims = (hidden_dim, hidden_dim, hidden_dim)
        if branch_hidden_layers is None:
            branch_hidden_layers = (hidden_layers, hidden_layers, hidden_layers)
        if len(branch_hidden_dims) != 3 or len(branch_hidden_layers) != 3:
            raise ValueError("branch_hidden_dims and branch_hidden_layers must contain 3 values")

        self.branch_hidden_dims = tuple(int(v) for v in branch_hidden_dims)
        self.branch_hidden_layers = tuple(int(v) for v in branch_hidden_layers)

        self.low_branch = SmoothMLP(
            input_dim, self.branch_hidden_dims[0], feature_dim, self.branch_hidden_layers[0]
        )
        self.mid_branch = SmoothMLP(
            input_dim, self.branch_hidden_dims[1], feature_dim, self.branch_hidden_layers[1]
        )
        self.high_branch = SmoothMLP(
            input_dim, self.branch_hidden_dims[2], feature_dim, self.branch_hidden_layers[2]
        )

        self.scale_logits = nn.Parameter(torch.zeros(3))
        self.residual_fusion = ResidualFusion(3 * feature_dim, fusion_dim)
        self.output_layer = nn.Linear(fusion_dim, output_dim)

    def get_scale_weights(self):
        return F.softmax(self.scale_logits, dim=0)

    def get_frequencies(self):
        """Compatibility helper: this architecture uses no Fourier frequencies."""
        return torch.empty(0, device=self.scale_logits.device)

    def forward(self, x):
        weights = self.get_scale_weights()
        low = weights[0] * self.low_branch(x)
        mid = weights[1] * self.mid_branch(x)
        high = weights[2] * self.high_branch(x)
        fused = torch.cat([low, mid, high], dim=-1)
        return self.output_layer(self.residual_fusion(fused))
