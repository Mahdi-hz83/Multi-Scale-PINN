import os
import sys
import csv
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(1)
from scipy.integrate import solve_ivp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.pinn import PINN
from src.equations.burgers import DEFAULT_NU, initial_condition, residual, derivatives
from src.evaluation.metrics import mse, mae, relative_l2_error, max_absolute_error


def make_training_data(seed, n_ic=100, n_bc=200, n_pde=2000):
    g = torch.Generator().manual_seed(seed)

    x_ic = -1.0 + 2.0 * torch.rand((n_ic, 1), generator=g)
    t_ic = torch.zeros_like(x_ic)
    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    u_ic = initial_condition(x_ic)

    n_each = n_bc // 2
    t_left = torch.rand((n_each, 1), generator=g)
    t_right = torch.rand((n_bc - n_each, 1), generator=g)
    xt_left = torch.cat([-torch.ones_like(t_left), t_left], dim=1)
    xt_right = torch.cat([torch.ones_like(t_right), t_right], dim=1)
    xt_bc = torch.cat([xt_left, xt_right], dim=0)
    u_bc = torch.zeros((n_bc, 1), dtype=torch.float32)

    x_pde = -1.0 + 2.0 * torch.rand((n_pde, 1), generator=g)
    t_pde = torch.rand((n_pde, 1), generator=g)
    xt_pde = torch.cat([x_pde, t_pde], dim=1)

    return xt_ic, u_ic, xt_bc, u_bc, xt_pde


def generate_reference(nx=256, nt=101, nu=DEFAULT_NU):
    """Generate a deterministic finite-volume reference solution.

    A local Lax-Friedrichs/Rusanov flux handles the nonlinear convection term,
    while the viscous term uses a centered second derivative. Time integration
    uses RK4 with a conservative fixed step, and Dirichlet boundaries are
    enforced at every stage.
    """
    x = np.linspace(-1.0, 1.0, nx)
    t_out = np.linspace(0.0, 1.0, nt)
    dx = x[1] - x[0]
    u = -np.sin(np.pi * x).astype(np.float64)
    u[0] = 0.0
    u[-1] = 0.0

    # Conservative step satisfying both advective and viscous restrictions.
    dt = 5.0e-4

    def rhs(v):
        vv = v.copy()
        vv[0] = 0.0
        vv[-1] = 0.0
        f = 0.5 * vv * vv
        alpha = np.maximum(np.abs(vv[:-1]), np.abs(vv[1:]))
        flux = 0.5 * (f[:-1] + f[1:]) - 0.5 * alpha * (vv[1:] - vv[:-1])
        dv = np.zeros_like(vv)
        dv[1:-1] = -(flux[1:] - flux[:-1]) / dx
        dv[1:-1] += nu * (vv[2:] - 2.0 * vv[1:-1] + vv[:-2]) / (dx * dx)
        return dv

    snapshots = np.zeros((nt, nx), dtype=np.float64)
    snapshots[0] = u
    current_t = 0.0
    out_idx = 1

    while out_idx < nt:
        target_t = t_out[out_idx]
        h = min(dt, target_t - current_t)
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * h * k1)
        k3 = rhs(u + 0.5 * h * k2)
        k4 = rhs(u + h * k3)
        u = u + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        u[0] = 0.0
        u[-1] = 0.0
        current_t += h
        if current_t >= target_t - 1e-12:
            snapshots[out_idx] = u
            out_idx += 1

    return x, t_out, snapshots


def train_burgers(
    model,
    xt_ic,
    u_ic,
    xt_bc,
    u_bc,
    xt_pde,
    epochs=4000,
    learning_rate=1e-3,
    print_every=500,
    lambda_ic=1.0,
    lambda_bc=1.0,
    lambda_pde=1.0,
    nu=DEFAULT_NU,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        pred_ic = model(xt_ic)
        pred_bc = model(xt_bc)
        r = residual(model, xt_pde, nu=nu)

        loss_ic = torch.mean((pred_ic - u_ic) ** 2)
        loss_bc = torch.mean((pred_bc - u_bc) ** 2)
        loss_pde = torch.mean(r ** 2)
        total = lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_pde * loss_pde

        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite total loss at epoch {epoch}")

        total.backward()
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError(f"Non-finite gradient at epoch {epoch}")
        optimizer.step()

        row = (epoch, total.item(), loss_ic.item(), loss_bc.item(), loss_pde.item())
        history.append(row)
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:5d} | Total: {row[1]:.6e} | "
                f"IC: {row[2]:.6e} | BC: {row[3]:.6e} | PDE: {row[4]:.6e}"
            )

    return history, time.perf_counter() - start


def main(epochs=4000, short_test=False):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Architecture configuration is kept local to the experiment, following
    # the same pattern used by the existing decay/oscillator scripts.
    input_dim = 2
    hidden_dim = 32
    hidden_layers = 3
    output_dim = 1
    feature_dim = 8
    fusion_dim = 16
    branch_hidden_dims = (8, 16, 24)
    branch_hidden_layers = (1, 2, 3)
    n_ic = 100
    n_bc = 200
    n_pde = 2000
    learning_rate = 1e-3
    print_every = 500 if epochs >= 500 else max(1, epochs // 5)
    nu = DEFAULT_NU

    results_dir = os.path.join(PROJECT_ROOT, "results")
    raw_dir = os.path.join(results_dir, "raw")
    figure_dir = os.path.join(results_dir, "figures")
    table_dir = os.path.join(results_dir, "tables")
    log_dir = os.path.join(results_dir, "logs")
    for path in (raw_dir, figure_dir, table_dir, log_dir):
        os.makedirs(path, exist_ok=True)

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

    xt_ic, u_ic, xt_bc, u_bc, xt_pde = make_training_data(seed, n_ic, n_bc, n_pde)

    # Explicit pre-training autograd/PDE smoke test.
    test_xt = torch.tensor([[0.0, 0.5], [0.4, 0.25], [-0.7, 0.8]], dtype=torch.float32)
    u, ux, ut, uxx = derivatives(model, test_xt)
    r0 = residual(model, test_xt, nu=nu)
    for name, value in {"u": u, "u_x": ux, "u_t": ut, "u_xx": uxx, "residual": r0}.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite {name} in pre-training smoke test")

    print("=" * 72)
    print("Final MSF-PINN Architecture (No Fourier) - Viscous Burgers Equation")
    print("=" * 72)
    print(f"nu:                  {nu:.15f}")
    print("Domain:              x in [-1,1], t in [0,1]")
    print(f"Seed:                {seed}")
    print(f"Epochs:              {epochs}")
    print(f"IC / BC / PDE:       {n_ic} / {n_bc} / {n_pde}")
    print(f"Learning Rate:       {learning_rate}")
    print(f"Branch Widths:       {model.branch_hidden_dims}")
    print(f"Branch Depths:       {model.branch_hidden_layers}")
    print(f"Parameters:          {sum(p.numel() for p in model.parameters())}")
    print("Pre-test derivatives: finite")
    print("=" * 72)

    history, training_time = train_burgers(
        model,
        xt_ic,
        u_ic,
        xt_bc,
        u_bc,
        xt_pde,
        epochs=epochs,
        learning_rate=learning_rate,
        print_every=print_every,
        nu=nu,
    )

    suffix = "_short" if short_test else ""
    checkpoint_path = os.path.join(raw_dir, f"burgers_model_checkpoint{suffix}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": model.architecture_info(),
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "nu": nu,
            "n_ic": n_ic,
            "n_bc": n_bc,
            "n_pde": n_pde,
        },
        checkpoint_path,
    )

    # Checkpoint load validation.
    reloaded = PINN(
        input_dim=input_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers,
        output_dim=output_dim, feature_dim=feature_dim, fusion_dim=fusion_dim,
        branch_hidden_dims=branch_hidden_dims, branch_hidden_layers=branch_hidden_layers,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        reload_diff = torch.max(torch.abs(model(test_xt) - reloaded(test_xt))).item()

    loss_csv = os.path.join(raw_dir, f"burgers_loss_history{suffix}.csv")
    with open(loss_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total_loss", "ic_loss", "bc_loss", "pde_loss"])
        writer.writerows(history)

    # Evaluation against a deterministic numerical reference.
    x_ref, t_ref, u_ref_np = generate_reference(nx=256, nt=101, nu=nu)
    X, T = np.meshgrid(x_ref, t_ref)
    xt_eval = torch.tensor(np.column_stack([X.ravel(), T.ravel()]), dtype=torch.float32)
    with torch.no_grad():
        u_pred = model(xt_eval).reshape(len(t_ref), len(x_ref)).cpu()
    u_ref = torch.tensor(u_ref_np, dtype=torch.float32)

    metrics = {
        "mse": mse(u_pred, u_ref),
        "mae": mae(u_pred, u_ref),
        "relative_l2_error": relative_l2_error(u_pred, u_ref),
        "max_absolute_error": max_absolute_error(u_pred, u_ref),
    }

    pred_csv = os.path.join(raw_dir, f"burgers_prediction{suffix}.csv")
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "t", "u_reference", "u_prediction", "absolute_error"])
        for xv, tv, ur, up in zip(X.ravel(), T.ravel(), u_ref_np.ravel(), u_pred.numpy().ravel()):
            writer.writerow([xv, tv, ur, up, abs(up - ur)])

    result_csv = os.path.join(table_dir, f"burgers_results{suffix}.csv")
    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "problem", "epochs", "nu", "parameters", "training_time_seconds",
            "final_loss", "mse", "mae", "relative_l2_error", "max_absolute_error",
            "reload_max_abs_diff", "final_frequencies", "final_scale_weights",
        ])
        writer.writerow([
            model.model_type, "Viscous Burgers Equation", epochs, nu,
            sum(p.numel() for p in model.parameters()), training_time, history[-1][1],
            metrics["mse"], metrics["mae"], metrics["relative_l2_error"],
            metrics["max_absolute_error"], reload_diff,
            model.get_frequencies().detach().cpu().tolist(),
            model.get_scale_weights().detach().cpu().tolist(),
        ])

    log_path = os.path.join(log_dir, f"burgers_training{suffix}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Final MSF-PINN Architecture - Burgers Training Log\n")
        f.write("=" * 72 + "\n")
        f.write("Equation: u_t + u*u_x - nu*u_xx = 0\n")
        f.write(f"nu: {nu:.15f}\n")
        f.write("Domain: x in [-1,1], t in [0,1]\n")
        f.write("IC: u(x,0) = -sin(pi*x)\n")
        f.write("BC: u(-1,t)=u(1,t)=0\n")
        f.write(f"Seed: {seed}\nEpochs: {epochs}\n")
        f.write(f"IC/BC/PDE points: {n_ic}/{n_bc}/{n_pde}\n")
        f.write(f"Architecture: {model.architecture_info()}\n")
        f.write(f"Parameters: {sum(p.numel() for p in model.parameters())}\n")
        f.write(f"Training time: {training_time:.6f} s\n")
        f.write(f"Final loss: {history[-1][1]:.12e}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.12e}\n")
        f.write(f"Checkpoint reload max abs diff: {reload_diff:.12e}\n")
        f.write(f"Final frequencies: {model.get_frequencies().detach().cpu().tolist()}\n")
        f.write(f"Final scale weights: {model.get_scale_weights().detach().cpu().tolist()}\n")

    # Loss plot
    plt.figure(figsize=(8, 5))
    h = np.asarray(history)
    plt.semilogy(h[:, 0], h[:, 1], label="Total")
    plt.semilogy(h[:, 0], h[:, 2], label="IC")
    plt.semilogy(h[:, 0], h[:, 3], label="BC")
    plt.semilogy(h[:, 0], h[:, 4], label="PDE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Burgers PINN Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f"burgers_loss{suffix}.png"), dpi=180)
    plt.close()

    # Prediction/reference/error heatmaps saved as separate figures.
    for name, data, title in [
        ("prediction", u_pred.numpy(), "Burgers PINN Prediction"),
        ("reference", u_ref_np, "Burgers Numerical Reference"),
        ("error", np.abs(u_pred.numpy() - u_ref_np), "Burgers Absolute Error"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.imshow(data, extent=[-1, 1, 0, 1], origin="lower", aspect="auto")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, f"burgers_{name}{suffix}.png"), dpi=180)
        plt.close()

    print("\nBurgers metrics")
    for k, v in metrics.items():
        print(f"{k:22s}: {v:.8e}")
    print(f"Training time          : {training_time:.3f} s")
    print(f"Checkpoint reload diff : {reload_diff:.8e}")
    print(f"Checkpoint             : {checkpoint_path}")
    return metrics


if __name__ == "__main__":
    main()
