# ============================================================
# Three-Scale Physics-Informed Neural Network
# ============================================================
#
# Architecture:
#
#                         Input x
#                            |
#              +-------------+-------------+
#              |             |             |
#              v             v             v
#         Low-Scale      Mid-Scale      High-Scale
#         Encoding       Encoding        Encoding
#              |             |             |
#              v             v             v
#           MLP-L          MLP-M         MLP-H
#              |             |             |
#              +-------------+-------------+
#                            |
#                       Feature Fusion
#                            |
#                        Fusion MLP
#                            |
#                            v
#                           u(x)
#
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# Activation
# ============================================================

class TanhMLP(nn.Module):
    """
    Simple MLP block using Tanh activation.

    Tanh is commonly used in PINNs because it is smooth
    and provides stable higher-order derivatives.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_layers=2,
    ):
        super().__init__()

        layers = []

        # First layer
        layers.append(
            nn.Linear(
                input_dim,
                hidden_dim,
            )
        )

        layers.append(
            nn.Tanh()
        )

        # Additional hidden layers
        for _ in range(hidden_layers - 1):

            layers.append(
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                )
            )

            layers.append(
                nn.Tanh()
            )

        # Output feature layer
        layers.append(
            nn.Linear(
                hidden_dim,
                output_dim,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Three-Scale PINN
# ============================================================

class MultiScalePINN(nn.Module):
    """
    Three-Scale Physics-Informed Neural Network.

    The network contains three parallel branches:

        1. Low-frequency branch
        2. Mid-frequency branch
        3. High-frequency branch

    Each branch receives a different sinusoidal encoding
    of the input. The extracted features are concatenated
    and passed through a fusion network.
    """

    def __init__(
        self,
        input_dim=1,
        hidden_dim=16,
        feature_dim=16,
        fusion_hidden_dim=16,
        output_dim=1,
        hidden_layers=2,
        low_frequency=1.0,
        mid_frequency=3.0,
        high_frequency=10.0,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Store configuration
        # ----------------------------------------------------

        self.input_dim = input_dim

        self.hidden_dim = hidden_dim

        self.feature_dim = feature_dim

        self.fusion_hidden_dim = fusion_hidden_dim

        self.output_dim = output_dim

        self.hidden_layers = hidden_layers

        self.low_frequency = low_frequency

        self.mid_frequency = mid_frequency

        self.high_frequency = high_frequency

        # ----------------------------------------------------
        # Each sinusoidal encoding contains:
        #
        # sin(w*x)
        # cos(w*x)
        #
        # Therefore:
        #
        # encoded_dim = 2 * input_dim
        # ----------------------------------------------------

        encoded_dim = 2 * input_dim

        # ----------------------------------------------------
        # Low-frequency branch
        # ----------------------------------------------------

        self.low_branch = TanhMLP(
            input_dim=encoded_dim,
            hidden_dim=hidden_dim,
            output_dim=feature_dim,
            hidden_layers=hidden_layers,
        )

        # ----------------------------------------------------
        # Mid-frequency branch
        # ----------------------------------------------------

        self.mid_branch = TanhMLP(
            input_dim=encoded_dim,
            hidden_dim=hidden_dim,
            output_dim=feature_dim,
            hidden_layers=hidden_layers,
        )

        # ----------------------------------------------------
        # High-frequency branch
        # ----------------------------------------------------

        self.high_branch = TanhMLP(
            input_dim=encoded_dim,
            hidden_dim=hidden_dim,
            output_dim=feature_dim,
            hidden_layers=hidden_layers,
        )

        # ----------------------------------------------------
        # Feature Fusion
        #
        # Three feature vectors are concatenated:
        #
        # [low | mid | high]
        #
        # Dimension:
        #
        # 3 * feature_dim
        # ----------------------------------------------------

        fusion_input_dim = (
            3 * feature_dim
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                fusion_input_dim,
                fusion_hidden_dim,
            ),

            nn.Tanh(),

            nn.Linear(
                fusion_hidden_dim,
                fusion_hidden_dim,
            ),

            nn.Tanh(),
        )

        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

        self.output_layer = nn.Linear(
            fusion_hidden_dim,
            output_dim,
        )

    # ========================================================
    # Frequency Encoding
    # ========================================================

    def encode(
        self,
        x,
        frequency,
    ):
        """
        Apply sinusoidal frequency encoding.

        For every input dimension x:

            sin(f * x)
            cos(f * x)

        are generated.

        This provides an explicit frequency scale
        for each branch.
        """

        sin_features = torch.sin(
            frequency * x
        )

        cos_features = torch.cos(
            frequency * x
        )

        encoded = torch.cat(
            [
                sin_features,
                cos_features,
            ],
            dim=-1,
        )

        return encoded

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):
        """
        Forward pass through the three-scale architecture.
        """

        # ----------------------------------------------------
        # Low-frequency representation
        # ----------------------------------------------------

        low_input = self.encode(
            x,
            self.low_frequency,
        )

        low_features = self.low_branch(
            low_input
        )

        # ----------------------------------------------------
        # Mid-frequency representation
        # ----------------------------------------------------

        mid_input = self.encode(
            x,
            self.mid_frequency,
        )

        mid_features = self.mid_branch(
            mid_input
        )

        # ----------------------------------------------------
        # High-frequency representation
        # ----------------------------------------------------

        high_input = self.encode(
            x,
            self.high_frequency,
        )

        high_features = self.high_branch(
            high_input
        )

        # ----------------------------------------------------
        # Feature Fusion
        # ----------------------------------------------------

        fused_features = torch.cat(
            [
                low_features,
                mid_features,
                high_features,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # Fusion Network
        # ----------------------------------------------------

        fused_output = self.fusion(
            fused_features
        )

        # ----------------------------------------------------
        # Final Output
        # ----------------------------------------------------

        output = self.output_layer(
            fused_output
        )

        return output