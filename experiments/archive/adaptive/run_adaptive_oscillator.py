# ============================================================
# Adaptive Three-Scale PINN
# Harmonic Oscillator ODE Experiment
# ============================================================
#
# Equation:
#
#       u''(x) + u(x) = 0
#
# Initial conditions:
#
#       u(0)  = 0
#       u'(0) = 1
#
# Exact solution:
#
#       u(x) = sin(x)
#
# ============================================================

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


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
# Imports
# ============================================================

from src.models.adaptive_multiscale_pinn import (
    AdaptiveMultiScalePINN
)

from src.equations.ode_oscillator import (
    residual as oscillator_residual
)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Configuration
# ============================================================

EPOCHS = 5000
COLLOCATION_POINTS = 100
LEARNING_RATE = 1e-3

HIDDEN_DIM = 16
FEATURE_DIM = 16
FUSION_DIM = 16
HIDDEN_LAYERS = 2

LOW_FREQUENCY = 1.0
MID_FREQUENCY = 3.0
HIGH_FREQUENCY = 10.0

X_MIN = 0.0
X_MAX = 2.0

PRINT_EVERY = 500


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
)

RAW_DIR = os.path.join(
    RESULTS_DIR,
    "raw",
)

FIGURES_DIR = os.path.join(
    RESULTS_DIR,
    "figures",
)

TABLES_DIR = os.path.join(
    RESULTS_DIR,
    "tables",
)

LOGS_DIR = os.path.join(
    RESULTS_DIR,
    "logs",
)


# ============================================================
# Create Directories
# ============================================================

for directory in [
    RAW_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    LOGS_DIR,
]:
    os.makedirs(
        directory,
        exist_ok=True,
    )


# ============================================================
# Exact Solution
# ============================================================

def exact_solution(x):
    return torch.sin(x)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    prediction,
    exact,
):
    error = prediction - exact

    mse = torch.mean(
        error ** 2
    ).item()

    mae = torch.mean(
        torch.abs(error)
    ).item()

    relative_l2 = (
        torch.linalg.norm(error)
        /
        torch.linalg.norm(exact)
    ).item()

    maximum_error = torch.max(
        torch.abs(error)
    ).item()

    return (
        mse,
        mae,
        relative_l2,
        maximum_error,
    )


# ============================================================
# Initial Derivative Loss
# ============================================================

def compute_initial_derivative_loss(
    model,
    x0,
    target_derivative,
):
    """
    Compute:

        (u'(0) - 1)^2
    """

    x = x0.clone().detach()
    x.requires_grad_(True)

    prediction = model(x)

    derivative = torch.autograd.grad(
        outputs=prediction,
        inputs=x,
        grad_outputs=torch.ones_like(
            prediction
        ),
        create_graph=True,
        retain_graph=True,
    )[0]

    loss = torch.mean(
        (
            derivative
            -
            target_derivative
        ) ** 2
    )

    return loss


# ============================================================
# Training
# ============================================================

def train(
    model,
    x_collocation,
    x0,
    u0,
    derivative0,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_history = []

    start_time = time.perf_counter()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Physics Loss
        # ----------------------------------------------------

        residual = oscillator_residual(
            model,
            x_collocation,
        )

        physics_loss = torch.mean(
            residual ** 2
        )

        # ----------------------------------------------------
        # Initial Value Loss
        # ----------------------------------------------------

        prediction_x0 = model(
            x0
        )

        initial_loss = torch.mean(
            (
                prediction_x0
                -
                u0
            ) ** 2
        )

        # ----------------------------------------------------
        # Initial Derivative Loss
        # ----------------------------------------------------

        derivative_loss = (
            compute_initial_derivative_loss(
                model=model,
                x0=x0,
                target_derivative=derivative0,
            )
        )

        # ----------------------------------------------------
        # Total Loss
        # ----------------------------------------------------

        total_loss = (
            physics_loss
            +
            initial_loss
            +
            derivative_loss
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        total_loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Store History
        # ----------------------------------------------------

        loss_value = (
            total_loss
            .detach()
            .item()
        )

        loss_history.append(
            loss_value
        )

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % PRINT_EVERY == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch {epoch:5d} | "
                f"Total Loss: "
                f"{total_loss.item():.6e} | "
                f"Physics: "
                f"{physics_loss.item():.6e} | "
                f"IC: "
                f"{initial_loss.item():.6e} | "
                f"IC Derivative: "
                f"{derivative_loss.item():.6e}"
            )

    training_time = (
        time.perf_counter()
        -
        start_time
    )

    return (
        model,
        loss_history,
        training_time,
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "Adaptive Three-Scale PINN "
        "- Harmonic Oscillator ODE"
    )
    print("=" * 70)

    print(
        "Model:               "
        "Adaptive 3S-PINN"
    )

    print(
        "Problem:             "
        "Harmonic Oscillator ODE"
    )

    print(
        "Equation:            "
        "u''(x) + u(x) = 0"
    )

    print(
        "Initial Condition:   "
        "u(0) = 0"
    )

    print(
        "Derivative Condition:"
        "u'(0) = 1"
    )

    print(
        "Exact Solution:      "
        "u(x) = sin(x)"
    )

    print(
        f"Domain:              "
        f"[{X_MIN}, {X_MAX}]"
    )

    print(
        f"Seed:                {SEED}"
    )

    print(
        f"Epochs:              {EPOCHS}"
    )

    print(
        f"Collocation Points:  "
        f"{COLLOCATION_POINTS}"
    )

    print(
        f"Learning Rate:       "
        f"{LEARNING_RATE}"
    )

    print(
        f"Hidden Dimension:    "
        f"{HIDDEN_DIM}"
    )

    print(
        f"Feature Dimension:   "
        f"{FEATURE_DIM}"
    )

    print(
        f"Fusion Dimension:    "
        f"{FUSION_DIM}"
    )

    print(
        f"Hidden Layers:       "
        f"{HIDDEN_LAYERS}"
    )

    print(
        f"Low Frequency:       "
        f"{LOW_FREQUENCY}"
    )

    print(
        f"Mid Frequency:       "
        f"{MID_FREQUENCY}"
    )

    print(
        f"High Frequency:      "
        f"{HIGH_FREQUENCY}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AdaptiveMultiScalePINN(
        input_dim=1,
        hidden_dim=HIDDEN_DIM,
        feature_dim=FEATURE_DIM,
        fusion_hidden_dim=FUSION_DIM,
        output_dim=1,
        hidden_layers=HIDDEN_LAYERS,
        low_frequency=LOW_FREQUENCY,
        mid_frequency=MID_FREQUENCY,
        high_frequency=HIGH_FREQUENCY,
    )

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters:          "
        f"{number_of_parameters}"
    )

    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Collocation Points
    # --------------------------------------------------------

    x_collocation = torch.linspace(
        X_MIN,
        X_MAX,
        COLLOCATION_POINTS,
    ).reshape(
        -1,
        1,
    )

    x_collocation.requires_grad_(True)

    # --------------------------------------------------------
    # Initial Conditions
    # --------------------------------------------------------

    x0 = torch.tensor(
        [[0.0]],
        dtype=torch.float32,
    )

    u0 = torch.tensor(
        [[0.0]],
        dtype=torch.float32,
    )

    derivative0 = torch.tensor(
        [[1.0]],
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    (
        model,
        loss_history,
        training_time,
    ) = train(
        model=model,
        x_collocation=x_collocation,
        x0=x0,
        u0=u0,
        derivative0=derivative0,
    )

    # ========================================================
    # Evaluation
    # ========================================================

    model.eval()

    x_test = torch.linspace(
        X_MIN,
        X_MAX,
        1000,
    ).reshape(
        -1,
        1,
    )

    with torch.no_grad():

        prediction = model(
            x_test
        )

        exact = exact_solution(
            x_test
        )

    (
        mse,
        mae,
        relative_l2,
        maximum_error,
    ) = compute_metrics(
        prediction,
        exact,
    )

    final_loss = loss_history[-1]

    minimum_loss = min(
        loss_history
    )

    minimum_loss_epoch = (
        np.argmin(
            loss_history
        )
        + 1
    )

    # --------------------------------------------------------
    # Learned Scale Weights
    # --------------------------------------------------------

    scale_weights = (
        model
        .get_scale_weights()
        .detach()
        .cpu()
        .numpy()
    )

    low_weight = scale_weights[0]
    mid_weight = scale_weights[1]
    high_weight = scale_weights[2]

    # ========================================================
    # Final Results
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Final Evaluation Results"
    )
    print("=" * 70)

    print(
        "Model:               "
        "Adaptive 3S-PINN"
    )

    print(
        f"Parameters:          "
        f"{number_of_parameters}"
    )

    print(
        f"Final Loss:          "
        f"{final_loss:.6e}"
    )

    print(
        f"Minimum Loss:        "
        f"{minimum_loss:.6e}"
    )

    print(
        f"Minimum Loss Epoch:  "
        f"{minimum_loss_epoch}"
    )

    print(
        f"MSE:                 "
        f"{mse:.6e}"
    )

    print(
        f"MAE:                 "
        f"{mae:.6e}"
    )

    print(
        f"Relative L2 Error:   "
        f"{relative_l2:.6e}"
    )

    print(
        f"Maximum Error:       "
        f"{maximum_error:.6e}"
    )

    print(
        f"Training Time:       "
        f"{training_time:.4f} seconds"
    )

    print()
    print(
        "Learned Scale Weights"
    )

    print("-" * 70)

    print(
        f"Low Scale Weight:    "
        f"{low_weight:.6f}"
    )

    print(
        f"Mid Scale Weight:    "
        f"{mid_weight:.6f}"
    )

    print(
        f"High Scale Weight:   "
        f"{high_weight:.6f}"
    )

    print(
        f"Weight Sum:          "
        f"{(
            low_weight
            +
            mid_weight
            +
            high_weight
        ):.6f}"
    )

    print("=" * 70)

    # ========================================================
    # Save Prediction Data
    # ========================================================

    prediction_path = os.path.join(
        RAW_DIR,
        "adaptive_oscillator_prediction.csv",
    )

    prediction_dataframe = pd.DataFrame(
        {
            "x":
                x_test.numpy().flatten(),

            "exact":
                exact.numpy().flatten(),

            "prediction":
                prediction.numpy().flatten(),

            "absolute_error":
                torch.abs(
                    prediction - exact
                )
                .numpy()
                .flatten(),
        }
    )

    prediction_dataframe.to_csv(
        prediction_path,
        index=False,
    )

    # ========================================================
    # Save Loss History
    # ========================================================

    loss_path = os.path.join(
        RAW_DIR,
        "adaptive_oscillator_loss_history.csv",
    )

    loss_dataframe = pd.DataFrame(
        {
            "epoch":
                np.arange(
                    1,
                    EPOCHS + 1,
                ),

            "loss":
                loss_history,
        }
    )

    loss_dataframe.to_csv(
        loss_path,
        index=False,
    )

    # ========================================================
    # Save Summary Table
    # ========================================================

    summary_path = os.path.join(
        TABLES_DIR,
        "adaptive_oscillator_results.csv",
    )

    summary_dataframe = pd.DataFrame(
        [
            {
                "model":
                    "Adaptive 3S-PINN",

                "problem":
                    "Harmonic Oscillator ODE",

                "equation":
                    "u'' + u = 0",

                "parameters":
                    number_of_parameters,

                "epochs":
                    EPOCHS,

                "collocation_points":
                    COLLOCATION_POINTS,

                "learning_rate":
                    LEARNING_RATE,

                "seed":
                    SEED,

                "final_loss":
                    final_loss,

                "minimum_loss":
                    minimum_loss,

                "minimum_loss_epoch":
                    minimum_loss_epoch,

                "mse":
                    mse,

                "mae":
                    mae,

                "relative_l2":
                    relative_l2,

                "maximum_error":
                    maximum_error,

                "training_time":
                    training_time,

                "low_scale_weight":
                    low_weight,

                "mid_scale_weight":
                    mid_weight,

                "high_scale_weight":
                    high_weight,
            }
        ]
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # Prediction Figure
    # ========================================================

    prediction_figure_path = os.path.join(
        FIGURES_DIR,
        "adaptive_oscillator_prediction.png",
    )

    if os.path.exists(
        prediction_figure_path
    ):
        os.remove(
            prediction_figure_path
        )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        x_test.numpy(),
        exact.numpy(),
        label="Exact Solution",
        linewidth=2,
    )

    plt.plot(
        x_test.numpy(),
        prediction.numpy(),
        "--",
        label="Adaptive 3S-PINN",
        linewidth=2,
    )

    plt.xlabel("x")
    plt.ylabel("u(x)")

    plt.title(
        "Adaptive 3S-PINN - "
        "Harmonic Oscillator"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        prediction_figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()

    # ========================================================
    # Loss Figure
    # ========================================================

    loss_figure_path = os.path.join(
        FIGURES_DIR,
        "adaptive_oscillator_loss.png",
    )

    if os.path.exists(
        loss_figure_path
    ):
        os.remove(
            loss_figure_path
        )

    plt.figure(
        figsize=(9, 6)
    )

    plt.semilogy(
        np.arange(
            1,
            EPOCHS + 1,
        ),
        loss_history,
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")

    plt.title(
        "Adaptive 3S-PINN - "
        "Training Loss"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        loss_figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()

    # ========================================================
    # Training Log
    # ========================================================

    log_path = os.path.join(
        LOGS_DIR,
        "adaptive_oscillator_training.txt",
    )

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as log_file:

        log_file.write(
            "Adaptive 3S-PINN "
            "Harmonic Oscillator ODE\n"
        )

        log_file.write(
            "=" * 70
            + "\n"
        )

        log_file.write(
            f"Seed: {SEED}\n"
        )

        log_file.write(
            f"Epochs: {EPOCHS}\n"
        )

        log_file.write(
            f"Collocation Points: "
            f"{COLLOCATION_POINTS}\n"
        )

        log_file.write(
            f"Learning Rate: "
            f"{LEARNING_RATE}\n"
        )

        log_file.write(
            f"Parameters: "
            f"{number_of_parameters}\n"
        )

        log_file.write(
            f"Final Loss: "
            f"{final_loss:.10e}\n"
        )

        log_file.write(
            f"Minimum Loss: "
            f"{minimum_loss:.10e}\n"
        )

        log_file.write(
            f"Minimum Loss Epoch: "
            f"{minimum_loss_epoch}\n"
        )

        log_file.write(
            f"MSE: "
            f"{mse:.10e}\n"
        )

        log_file.write(
            f"MAE: "
            f"{mae:.10e}\n"
        )

        log_file.write(
            f"Relative L2 Error: "
            f"{relative_l2:.10e}\n"
        )

        log_file.write(
            f"Maximum Error: "
            f"{maximum_error:.10e}\n"
        )

        log_file.write(
            f"Training Time: "
            f"{training_time:.6f}\n"
        )

        log_file.write("\n")

        log_file.write(
            "Learned Scale Weights\n"
        )

        log_file.write(
            f"Low:  "
            f"{low_weight:.10f}\n"
        )

        log_file.write(
            f"Mid:  "
            f"{mid_weight:.10f}\n"
        )

        log_file.write(
            f"High: "
            f"{high_weight:.10f}\n"
        )

    # ========================================================
    # Final Output
    # ========================================================

    print()
    print(
        "Prediction data saved to:"
    )
    print(
        prediction_path
    )

    print()
    print(
        "Loss history saved to:"
    )
    print(
        loss_path
    )

    print()
    print(
        "Summary table saved to:"
    )
    print(
        summary_path
    )

    print()
    print(
        "Prediction figure saved to:"
    )
    print(
        prediction_figure_path
    )

    print()
    print(
        "Loss figure saved to:"
    )
    print(
        loss_figure_path
    )

    print()
    print(
        "Training log saved to:"
    )
    print(
        log_path
    )

    print()
    print("=" * 70)
    print(
        "Adaptive 3S-PINN Harmonic "
        "Oscillator experiment completed successfully."
    )
    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()