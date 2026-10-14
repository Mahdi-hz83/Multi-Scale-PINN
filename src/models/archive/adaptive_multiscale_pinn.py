# ============================================================
# Adaptive Three-Scale Physics-Informed Neural Network
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
#                  Adaptive Scale Weighting
#                            |
#                     w_L  w_M  w_H
#                            |
#                            v
#                    Weighted Features
#                            |
#                      Fusion Network
#                            |
#                            v
#                           u(x)
#
# The three scale weights are learned automatically.
#
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Activation
# ============================================================

class TanhMLP(nn.Module):
    """
    Smooth MLP block based on Tanh activation.

    Tanh is appropriate for PINNs because it is smooth and
    supports stable first- and higher-order derivatives.
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

        # First hidden layer
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

        # Feature output layer
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
# Adaptive Multi-Scale PINN
# ============================================================

class AdaptiveMultiScalePINN(nn.Module):
    """
    Adaptive Three-Scale Physics-Informed Neural Network.

    The architecture contains three parallel branches:

        Low-frequency
        Mid-frequency
        High-frequency

    Each branch receives a different sinusoidal encoding.

    Unlike the basic MultiScalePINN, the feature contribution
    of each scale is controlled by learnable adaptive weights.

    The weights are normalized using Softmax:

        w_L + w_M + w_H = 1

    This allows the network to automatically determine the
    relative importance of different frequency scales.
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
        # Configuration
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
        # Sin/Cos encoding dimension
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
        # Adaptive Scale Weighting
        #
        # Three learnable logits are converted into normalized
        # scale weights through Softmax.
        #
        # Initial state:
        #
        # [1/3, 1/3, 1/3]
        #
        # The network learns how these weights should change.
        # ----------------------------------------------------

        self.scale_logits = nn.Parameter(
            torch.zeros(3)
        )

        # ----------------------------------------------------
        # Feature Fusion
        # ----------------------------------------------------
        #
        # Weighted features are concatenated rather than
        # simply concatenated without weighting.
        #
        # [wL * FL | wM * FM | wH * FH]
        #
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
        # Output Layer
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
        Sinusoidal frequency encoding.

        For each input dimension:

            sin(f*x)
            cos(f*x)
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
    # Scale Weights
    # ========================================================

    def get_scale_weights(self):
        """
        Return normalized adaptive scale weights.

        Returns
        -------
        torch.Tensor
            Tensor containing:

                [low_weight,
                 mid_weight,
                 high_weight]

        The values sum to one.
        """

        weights = F.softmax(
            self.scale_logits,
            dim=0,
        )

        return weights

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):
        """
        Forward pass through the adaptive multi-scale network.
        """

        # ----------------------------------------------------
        # Low-scale representation
        # ----------------------------------------------------

        low_input = self.encode(
            x,
            self.low_frequency,
        )

        low_features = self.low_branch(
            low_input
        )

        # ----------------------------------------------------
        # Mid-scale representation
        # ----------------------------------------------------

        mid_input = self.encode(
            x,
            self.mid_frequency,
        )

        mid_features = self.mid_branch(
            mid_input
        )

        # ----------------------------------------------------
        # High-scale representation
        # ----------------------------------------------------

        high_input = self.encode(
            x,
            self.high_frequency,
        )

        high_features = self.high_branch(
            high_input
        )

        # ----------------------------------------------------
        # Adaptive Scale Weights
        # ----------------------------------------------------

        weights = self.get_scale_weights()

        low_weight = weights[0]

        mid_weight = weights[1]

        high_weight = weights[2]

        # ----------------------------------------------------
        # Weighted Feature Representations
        # ----------------------------------------------------

        weighted_low = (
            low_weight * low_features
        )

        weighted_mid = (
            mid_weight * mid_features
        )

        weighted_high = (
            high_weight * high_features
        )

        # ----------------------------------------------------
        # Feature Fusion
        # ----------------------------------------------------

        fused_features = torch.cat(
            [
                weighted_low,
                weighted_mid,
                weighted_high,
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