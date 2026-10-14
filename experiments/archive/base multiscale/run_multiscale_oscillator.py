import os
import sys
import csv
import time

import matplotlib.pyplot as plt
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
# Imports
# ============================================================

from src.models.multiscale_pinn import MultiScalePINN

from src.equations.ode_oscillator import (
    residual,
    exact_solution,
)

from src.training.trainer import train

from src.evaluation.metrics import (
    mse,
    mae,
    relative_l2_error,
    max_absolute_error,
)


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # 1. Reproducibility
    # ========================================================

    seed = 42

    torch.manual_seed(seed)

    # ========================================================
    # 2. Configuration
    # ========================================================

    model_name = "3S-PINN"
    problem_name = "Harmonic Oscillator ODE"

    epochs = 5000
    learning_rate = 1e-3
    print_every = 500

    number_of_collocation_points = 100

    domain_start = 0.0
    domain_end = 2.0

    hidden_dim = 16
    feature_dim = 16
    fusion_hidden_dim = 16
    hidden_layers = 2

    low_frequency = 1.0
    mid_frequency = 3.0
    high_frequency = 10.0

    # ========================================================
    # 3. Results Directories
    # ========================================================

    results_dir = os.path.join(
        PROJECT_ROOT,
        "results",
    )

    raw_dir = os.path.join(
        results_dir,
        "raw",
    )

    figure_dir = os.path.join(
        results_dir,
        "figures",
    )

    table_dir = os.path.join(
        results_dir,
        "tables",
    )

    log_dir = os.path.join(
        results_dir,
        "logs",
    )

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ========================================================
    # 4. Output Files
    # ========================================================

    prediction_path = os.path.join(
        raw_dir,
        "multiscale_oscillator_prediction.csv",
    )

    loss_path = os.path.join(
        raw_dir,
        "multiscale_oscillator_loss_history.csv",
    )

    summary_path = os.path.join(
        table_dir,
        "multiscale_oscillator_results.csv",
    )

    prediction_figure_path = os.path.join(
        figure_dir,
        "multiscale_oscillator_prediction.png",
    )

    loss_figure_path = os.path.join(
        figure_dir,
        "multiscale_oscillator_loss.png",
    )

    log_path = os.path.join(
        log_dir,
        "multiscale_oscillator_training.txt",
    )

    # ========================================================
    # 5. Model
    # ========================================================

    model = MultiScalePINN(
        input_dim=1,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        output_dim=1,
        hidden_layers=hidden_layers,
        low_frequency=low_frequency,
        mid_frequency=mid_frequency,
        high_frequency=high_frequency,
    )

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    # ========================================================
    # 6. Collocation Points
    # ========================================================

    x_collocation = torch.linspace(
        domain_start,
        domain_end,
        number_of_collocation_points,
    ).reshape(-1, 1)

    # ========================================================
    # 7. Initial Conditions
    # ========================================================

    x0 = torch.tensor(
        [[0.0]],
        dtype=torch.float32,
    )

    u0 = torch.tensor(
        [[0.0]],
        dtype=torch.float32,
    )

    du0 = torch.tensor(
        [[1.0]],
        dtype=torch.float32,
    )

    # ========================================================
    # 8. Configuration Display
    # ========================================================

    print()
    print("=" * 70)
    print("Three-Scale PINN - Harmonic Oscillator ODE")
    print("=" * 70)

    print(f"Model:               {model_name}")
    print(f"Problem:             {problem_name}")
    print("Equation:            u''(x) + u(x) = 0")
    print("Initial Condition:   u(0) = 0")
    print("Derivative Condition:u'(0) = 1")
    print("Exact Solution:      u(x) = sin(x)")
    print(f"Domain:              [{domain_start}, {domain_end}]")
    print(f"Seed:                {seed}")
    print(f"Epochs:              {epochs}")
    print(
        f"Collocation Points:  "
        f"{number_of_collocation_points}"
    )
    print(f"Learning Rate:       {learning_rate}")
    print(f"Hidden Dimension:    {hidden_dim}")
    print(f"Feature Dimension:   {feature_dim}")
    print(f"Fusion Dimension:    {fusion_hidden_dim}")
    print(f"Hidden Layers:       {hidden_layers}")
    print(f"Low Frequency:       {low_frequency}")
    print(f"Mid Frequency:       {mid_frequency}")
    print(f"High Frequency:      {high_frequency}")
    print(f"Parameters:          {number_of_parameters}")

    print("=" * 70)
    print()

    # ========================================================
    # 9. Training
    # ========================================================

    start_time = time.perf_counter()

    (
        model,
        loss_history,
        trainer_time,
    ) = train(
        model=model,
        x_collocation=x_collocation,
        x0=x0,
        u0=u0,
        du0=du0,
        epochs=epochs,
        learning_rate=learning_rate,
        print_every=print_every,
        residual_function=residual,
    )

    training_time = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # 10. Evaluation Grid
    # ========================================================

    x_test = torch.linspace(
        domain_start,
        domain_end,
        500,
    ).reshape(-1, 1)

    # ========================================================
    # 11. Prediction
    # ========================================================

    model.eval()

    with torch.no_grad():
        prediction = model(x_test)

    # ========================================================
    # 12. Exact Solution
    # ========================================================

    exact = exact_solution(
        x_test
    )

    # ========================================================
    # 13. Metrics
    # ========================================================

    mse_value = mse(
        prediction,
        exact,
    )

    mae_value = mae(
        prediction,
        exact,
    )

    relative_l2 = relative_l2_error(
        prediction,
        exact,
    )

    max_error = max_absolute_error(
        prediction,
        exact,
    )

    # ========================================================
    # 14. Loss Statistics
    # ========================================================

    final_loss = loss_history[-1]

    minimum_loss = min(
        loss_history
    )

    minimum_loss_epoch = (
        loss_history.index(
            minimum_loss
        ) + 1
    )

    # ========================================================
    # 15. Final Results
    # ========================================================

    print()
    print("=" * 70)
    print("Final Evaluation Results")
    print("=" * 70)

    print(f"Model:               {model_name}")
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
        f"{mse_value:.6e}"
    )
    print(
        f"MAE:                 "
        f"{mae_value:.6e}"
    )
    print(
        f"Relative L2 Error:   "
        f"{relative_l2:.6e}"
    )
    print(
        f"Maximum Error:       "
        f"{max_error:.6e}"
    )
    print(
        f"Training Time:       "
        f"{training_time:.4f} seconds"
    )

    print("=" * 70)

    # ========================================================
    # 16. Save Prediction CSV
    # ========================================================

    with open(
        prediction_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "x",
                "exact_solution",
                "multiscale_prediction",
                "absolute_error",
            ]
        )

        for (
            x_value,
            exact_value,
            prediction_value,
        ) in zip(
            x_test.flatten().tolist(),
            exact.flatten().tolist(),
            prediction.flatten().tolist(),
        ):

            absolute_error = abs(
                exact_value
                - prediction_value
            )

            writer.writerow(
                [
                    x_value,
                    exact_value,
                    prediction_value,
                    absolute_error,
                ]
            )

    print()
    print("Prediction data saved to:")
    print(prediction_path)

    # ========================================================
    # 17. Save Loss History
    # ========================================================

    with open(
        loss_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "epoch",
                "total_loss",
            ]
        )

        for (
            epoch,
            loss_value,
        ) in enumerate(
            loss_history,
            start=1,
        ):

            writer.writerow(
                [
                    epoch,
                    loss_value,
                ]
            )

    print("Loss history saved to:")
    print(loss_path)

    # ========================================================
    # 18. Save Summary
    # ========================================================

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "model",
                "problem",
                "seed",
                "epochs",
                "collocation_points",
                "domain_start",
                "domain_end",
                "hidden_dim",
                "feature_dim",
                "fusion_hidden_dim",
                "hidden_layers",
                "low_frequency",
                "mid_frequency",
                "high_frequency",
                "learning_rate",
                "parameters",
                "final_loss",
                "minimum_loss",
                "minimum_loss_epoch",
                "mse",
                "mae",
                "relative_l2_error",
                "max_absolute_error",
                "training_time_seconds",
            ]
        )

        writer.writerow(
            [
                model_name,
                problem_name,
                seed,
                epochs,
                number_of_collocation_points,
                domain_start,
                domain_end,
                hidden_dim,
                feature_dim,
                fusion_hidden_dim,
                hidden_layers,
                low_frequency,
                mid_frequency,
                high_frequency,
                learning_rate,
                number_of_parameters,
                final_loss,
                minimum_loss,
                minimum_loss_epoch,
                mse_value,
                mae_value,
                relative_l2,
                max_error,
                training_time,
            ]
        )

    print("Summary table saved to:")
    print(summary_path)

    # ========================================================
    # 19. Prediction Figure
    # ========================================================

    plt.figure(
        figsize=(8, 5)
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
        label="3S-PINN Prediction",
        linewidth=2,
    )

    plt.xlabel("x")
    plt.ylabel("u(x)")

    plt.title(
        "3S-PINN vs Exact Solution - Harmonic Oscillator"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        prediction_figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print()
    print("Prediction figure saved to:")
    print(prediction_figure_path)

    # ========================================================
    # 20. Loss Figure
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.semilogy(
        range(
            1,
            len(loss_history) + 1,
        ),
        loss_history,
        label="3S-PINN Total Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "3S-PINN Training Loss - Harmonic Oscillator"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        loss_figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print("Loss figure saved to:")
    print(loss_figure_path)

    # ========================================================
    # 21. Training Log
    # ========================================================

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as log_file:

        log_file.write(
            "=" * 70 + "\n"
        )

        log_file.write(
            "Three-Scale PINN Training Log\n"
        )

        log_file.write(
            "=" * 70 + "\n"
        )

        log_file.write(
            f"Model: {model_name}\n"
        )

        log_file.write(
            f"Problem: {problem_name}\n"
        )

        log_file.write(
            "Equation: u''(x) + u(x) = 0\n"
        )

        log_file.write(
            "Initial Condition: u(0) = 0\n"
        )

        log_file.write(
            "Derivative Condition: u'(0) = 1\n"
        )

        log_file.write(
            "Exact Solution: u(x) = sin(x)\n"
        )

        log_file.write(
            f"Domain: [{domain_start}, {domain_end}]\n"
        )

        log_file.write(
            f"Seed: {seed}\n"
        )

        log_file.write(
            f"Epochs: {epochs}\n"
        )

        log_file.write(
            f"Collocation Points: "
            f"{number_of_collocation_points}\n"
        )

        log_file.write(
            f"Learning Rate: {learning_rate}\n"
        )

        log_file.write(
            f"Parameters: "
            f"{number_of_parameters}\n"
        )

        log_file.write(
            f"Low Frequency: {low_frequency}\n"
        )

        log_file.write(
            f"Mid Frequency: {mid_frequency}\n"
        )

        log_file.write(
            f"High Frequency: {high_frequency}\n"
        )

        log_file.write(
            f"Training Time: "
            f"{training_time:.6f} seconds\n"
        )

        log_file.write(
            "=" * 70 + "\n\n"
        )

        log_file.write(
            "Training Loss History\n"
        )

        log_file.write(
            "=" * 70 + "\n"
        )

        for (
            epoch,
            loss_value,
        ) in enumerate(
            loss_history,
            start=1,
        ):

            log_file.write(
                f"Epoch {epoch:5d} | "
                f"Total Loss: "
                f"{loss_value:.10e}\n"
            )

        log_file.write(
            "\n"
        )

        log_file.write(
            "=" * 70 + "\n"
        )

        log_file.write(
            "Final Evaluation Results\n"
        )

        log_file.write(
            "=" * 70 + "\n"
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
            f"{mse_value:.10e}\n"
        )

        log_file.write(
            f"MAE: "
            f"{mae_value:.10e}\n"
        )

        log_file.write(
            f"Relative L2 Error: "
            f"{relative_l2:.10e}\n"
        )

        log_file.write(
            f"Maximum Error: "
            f"{max_error:.10e}\n"
        )

        log_file.write(
            f"Training Time: "
            f"{training_time:.6f} seconds\n"
        )

        log_file.write(
            "=" * 70 + "\n"
        )

    print()
    print("Training log saved to:")
    print(log_path)

    # ========================================================
    # 22. Completion
    # ========================================================

    print()
    print("=" * 70)
    print(
        "3S-PINN Harmonic Oscillator experiment "
        "completed successfully."
    )
    print("=" * 70)

    print()
    print("Results directory:")
    print(results_dir)
    print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()