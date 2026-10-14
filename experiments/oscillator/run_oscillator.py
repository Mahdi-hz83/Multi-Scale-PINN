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
# Project Imports
# ============================================================

from src.models.pinn import PINN

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
# Main Experiment
# ============================================================

def main():

    # ========================================================
    # 1. Reproducibility
    # ========================================================

    seed = 42

    torch.manual_seed(seed)

    # ========================================================
    # 2. Experiment Configuration
    # ========================================================

    model_name = "Final MSF-PINN Architecture (No Fourier)"

    problem_name = "Harmonic Oscillator ODE"

    input_dim = 1

    hidden_dim = 32

    hidden_layers = 3

    output_dim = 1

    # Multi-Scale architecture configuration.
    # Kept in the existing experiment configuration section so the
    # project does not need a new configuration-loading mechanism.
    feature_dim = 8

    fusion_dim = 16

    branch_hidden_dims = (8, 16, 24)

    branch_hidden_layers = (1, 2, 3)

    epochs = 4000

    learning_rate = 1e-3

    number_of_collocation_points = 100

    print_every = 500

    # ========================================================
    # 3. Result Directories
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

    os.makedirs(
        raw_dir,
        exist_ok=True,
    )

    os.makedirs(
        figure_dir,
        exist_ok=True,
    )

    os.makedirs(
        table_dir,
        exist_ok=True,
    )

    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    # ========================================================
    # 4. Log File
    # ========================================================

    log_path = os.path.join(
        log_dir,
        "oscillator_training.txt",
    )

    # ========================================================
    # 5. Create Model
    # ========================================================

    model = PINN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
        feature_dim=feature_dim,
        fusion_dim=fusion_dim,
        branch_hidden_dims=branch_hidden_dims,
        branch_hidden_layers=branch_hidden_layers,
    )

    architecture_info = model.architecture_info()

    # ========================================================
    # 6. Collocation Points
    # ========================================================

    x_collocation = torch.linspace(
        0.0,
        2.0,
        number_of_collocation_points,
    ).reshape(-1, 1)

    # ========================================================
    # 7. Initial Conditions
    # ========================================================

    # u(0) = 0

    x0 = torch.tensor(
        [[0.0]],
        dtype=torch.float32,
    )

    u0 = torch.tensor(
        [[0.0]],
        dtype=torch.float32,
    )

    # u'(0) = 1

    du0 = torch.tensor(
        [[1.0]],
        dtype=torch.float32,
    )

    # ========================================================
    # 8. Print Experiment Configuration
    # ========================================================

    print()
    print("=" * 65)
    print("Multi-Scale PINN - Harmonic Oscillator ODE")
    print("=" * 65)

    print(
        f"Model:               {model_name}"
    )

    print(
        f"Problem:             {problem_name}"
    )

    print(
        f"Equation:            u''(x) + u(x) = 0"
    )

    print(
        f"Initial Condition:   u(0) = 0"
    )

    print(
        f"Derivative Condition:u'(0) = 1"
    )

    print(
        f"Exact Solution:      u(x) = sin(x)"
    )

    print(
        f"Seed:                {seed}"
    )

    print(
        f"Epochs:              {epochs}"
    )

    print(
        f"Collocation Points:  "
        f"{number_of_collocation_points}"
    )

    print(
        f"Learning Rate:       {learning_rate}"
    )

    print(
        f"Hidden Dimension:    {hidden_dim}"
    )

    print(
        f"Hidden Layers:       {hidden_layers}"
    )

    print(
        f"Multi-Scale Branches:{architecture_info['scales']}"
    )

    print(
        f"Branch Widths:       {architecture_info['branch_hidden_dims']}"
    )

    print(
        f"Branch Depths:       {architecture_info['branch_hidden_layers']}"
    )

    print(
        f"Initial Frequencies: {architecture_info['initial_frequencies']}"
    )

    print(
        f"Fusion:              {architecture_info['fusion']}"
    )

    print("=" * 65)
    print()

    # ========================================================
    # 9. Start Training
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

    final_frequencies = (
        model.get_frequencies()
        .detach()
        .cpu()
        .tolist()
    )

    final_scale_weights = (
        model.get_scale_weights()
        .detach()
        .cpu()
        .tolist()
    )

    # ========================================================
    # 10. Save Training Log
    # ========================================================

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as log_file:

        log_file.write(
            "=" * 65 + "\n"
        )

        log_file.write(
            "Adaptive Multi-Scale PINN Training Log\n"
        )

        log_file.write(
            "=" * 65 + "\n"
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
            f"Hidden Dimension: {hidden_dim}\n"
        )

        log_file.write(
            f"Hidden Layers: {hidden_layers}\n"
        )

        log_file.write(
            f"Multi-Scale Branches: {architecture_info['scales']}\n"
        )

        log_file.write(
            f"Branch Widths: {architecture_info['branch_hidden_dims']}\n"
        )

        log_file.write(
            f"Branch Depths: {architecture_info['branch_hidden_layers']}\n"
        )

        log_file.write(
            f"Activation: {architecture_info['activation']}\n"
        )

        log_file.write(
            f"Fusion: {architecture_info['fusion']}\n"
        )

        log_file.write(
            f"Trainable Frequencies: {architecture_info['trainable_frequencies']}\n"
        )

        log_file.write(
            f"Initial Frequencies: {architecture_info['initial_frequencies']}\n"
        )

        log_file.write(
            f"Final Frequencies: {tuple(final_frequencies)}\n"
        )

        log_file.write(
            f"Final Scale Weights: {tuple(final_scale_weights)}\n"
        )

        log_file.write(
            f"Training Time: "
            f"{training_time:.6f} seconds\n"
        )

        log_file.write(
            "=" * 65 + "\n\n"
        )

        log_file.write(
            "Training Loss History\n"
        )

        log_file.write(
            "=" * 65 + "\n"
        )

        for epoch, loss_value in enumerate(
            loss_history,
            start=1,
        ):

            log_file.write(
                f"Epoch {epoch:5d} | "
                f"Total Loss: "
                f"{loss_value:.10e}\n"
            )

    # ========================================================
    # 11. Evaluation Points
    # ========================================================

    x_test = torch.linspace(
        0.0,
        2.0,
        500,
    ).reshape(-1, 1)

    # ========================================================
    # 12. Model Prediction
    # ========================================================

    model.eval()

    with torch.no_grad():

        prediction = model(
            x_test
        )

    # ========================================================
    # 13. Exact Solution
    # ========================================================

    exact = exact_solution(
        x_test
    )

    # ========================================================
    # 14. Calculate Metrics
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
    # 15. Loss Statistics
    # ========================================================

    final_loss = loss_history[-1]

    minimum_loss = min(
        loss_history
    )

    # ========================================================
    # 16. Number of Parameters
    # ========================================================

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    # ========================================================
    # 17. Print Final Results
    # ========================================================

    print()
    print("=" * 65)
    print("Final Evaluation Results")
    print("=" * 65)

    print(
        f"Model:               {model_name}"
    )

    print(
        f"Problem:             {problem_name}"
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

    print("=" * 65)

    # ========================================================
    # 18. Save Prediction CSV
    # ========================================================

    prediction_path = os.path.join(
        raw_dir,
        "oscillator_prediction.csv",
    )

    with open(
        prediction_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "x",
                "exact_solution",
                "pinn_prediction",
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
    print(
        "Prediction data saved to:"
    )
    print(
        prediction_path
    )

    # ========================================================
    # 19. Save Loss History CSV
    # ========================================================

    loss_path = os.path.join(
        raw_dir,
        "oscillator_loss_history.csv",
    )

    with open(
        loss_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

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

    print(
        "Loss history saved to:"
    )
    print(
        loss_path
    )

    # ========================================================
    # 20. Save Summary Table
    # ========================================================

    summary_path = os.path.join(
        table_dir,
        "oscillator_results.csv",
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "model",
                "problem",
                "seed",
                "epochs",
                "collocation_points",
                "hidden_dim",
                "hidden_layers",
                "learning_rate",
                "parameters",
                "final_loss",
                "minimum_loss",
                "mse",
                "mae",
                "relative_l2_error",
                "max_absolute_error",
                "training_time_seconds",
                "multiscale_branches",
                "branch_widths",
                "branch_depths",
                "final_frequencies",
                "final_scale_weights",
                "fusion",
                "feature_dim",
                "fusion_dim",
                "trainable_frequencies",
            ]
        )

        writer.writerow(
            [
                model_name,
                problem_name,
                seed,
                epochs,
                number_of_collocation_points,
                hidden_dim,
                hidden_layers,
                learning_rate,
                number_of_parameters,
                final_loss,
                minimum_loss,
                mse_value,
                mae_value,
                relative_l2,
                max_error,
                training_time,
                architecture_info["scales"],
                architecture_info["branch_hidden_dims"],
                architecture_info["branch_hidden_layers"],
                tuple(final_frequencies),
                tuple(final_scale_weights),
                architecture_info["fusion"],
                architecture_info["feature_dim"],
                architecture_info["fusion_dim"],
                architecture_info["trainable_frequencies"],
            ]
        )

    print(
        "Summary table saved to:"
    )
    print(
        summary_path
    )

    # ========================================================
    # 20B. Save and Verify Model Checkpoint
    # ========================================================

    checkpoint_path = os.path.join(
        raw_dir,
        "oscillator_model_checkpoint.pt",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "architecture": architecture_info,
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "final_frequencies": final_frequencies,
            "final_scale_weights": final_scale_weights,
        },
        checkpoint_path,
    )

    checkpoint_model = PINN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
        feature_dim=feature_dim,
        fusion_dim=fusion_dim,
        branch_hidden_dims=branch_hidden_dims,
        branch_hidden_layers=branch_hidden_layers,
    )

    checkpoint_data = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    checkpoint_model.load_state_dict(
        checkpoint_data["model_state_dict"]
    )

    checkpoint_model.eval()

    with torch.no_grad():
        checkpoint_prediction = checkpoint_model(x_test)

    checkpoint_reload_max_difference = torch.max(
        torch.abs(checkpoint_prediction - prediction)
    ).item()

    print(
        "Model checkpoint saved to:"
    )
    print(
        checkpoint_path
    )
    print(
        f"Checkpoint reload max difference: "
        f"{checkpoint_reload_max_difference:.6e}"
    )

    # ========================================================
    # 21. Save Prediction Figure
    # ========================================================

    prediction_figure_path = os.path.join(
        figure_dir,
        "oscillator_prediction.png",
    )

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
        label="Multi-Scale PINN Prediction",
        linewidth=2,
    )

    plt.xlabel(
        "x"
    )

    plt.ylabel(
        "u(x)"
    )

    plt.title(
        "Multi-Scale PINN vs Exact Solution - Harmonic Oscillator"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        prediction_figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    print()
    print(
        "Prediction figure saved to:"
    )

    print(
        prediction_figure_path
    )

    plt.show()

    plt.close()

    # ========================================================
    # 22. Save Loss Figure
    # ========================================================

    loss_figure_path = os.path.join(
        figure_dir,
        "oscillator_loss.png",
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.semilogy(
        range(
            1,
            len(loss_history) + 1,
        ),
        loss_history,
        label="Total Loss",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Multi-Scale PINN Training Loss - Harmonic Oscillator"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        loss_figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        "Loss figure saved to:"
    )

    print(
        loss_figure_path
    )

    plt.show()

    plt.close()

    # ========================================================
    # 23. Append Final Results to Log
    # ========================================================

    with open(
        log_path,
        "a",
        encoding="utf-8",
    ) as log_file:

        log_file.write("\n")

        log_file.write(
            "=" * 65 + "\n"
        )

        log_file.write(
            "Final Evaluation Results\n"
        )

        log_file.write(
            "=" * 65 + "\n"
        )

        log_file.write(
            f"Model: {model_name}\n"
        )

        log_file.write(
            f"Problem: {problem_name}\n"
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
            f"Final Frequencies: {tuple(final_frequencies)}\n"
        )

        log_file.write(
            f"Final Scale Weights: {tuple(final_scale_weights)}\n"
        )

        log_file.write(
            f"Checkpoint: {checkpoint_path}\n"
        )

        log_file.write(
            f"Checkpoint Reload Max Difference: "
            f"{checkpoint_reload_max_difference:.10e}\n"
        )

        log_file.write(
            "=" * 65 + "\n"
        )

    # ========================================================
    # 24. Final Message
    # ========================================================

    print()
    print("=" * 65)
    print("Oscillator experiment completed successfully.")
    print("=" * 65)

    print()
    print(
        "Results directory:"
    )

    print(
        results_dir
    )

    print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()