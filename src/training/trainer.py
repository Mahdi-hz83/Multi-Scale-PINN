import torch

from src.equations.ode_decay import residual as decay_residual


def compute_initial_condition_loss(
    model,
    x0,
    u0,
):
    """
    Compute loss for the initial value condition:

        u(x0) = u0
    """

    prediction = model(x0)

    loss = torch.mean(
        (prediction - u0) ** 2
    )

    return loss


def compute_initial_derivative_loss(
    model,
    x0,
    du0,
):
    """
    Compute loss for the initial derivative condition:

        u'(x0) = du0
    """

    x0 = x0.clone().detach().requires_grad_(True)

    prediction = model(x0)

    du_dx = torch.autograd.grad(
        outputs=prediction,
        inputs=x0,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True,
        retain_graph=True,
    )[0]

    loss = torch.mean(
        (du_dx - du0) ** 2
    )

    return loss


def train(
    model,
    x_collocation,
    x0,
    u0,
    epochs=4000,
    learning_rate=1e-3,
    print_every=500,
    residual_function=None,
    du0=None,
    derivative_loss_weight=1.0,
    physics_loss_weight=1.0,
    initial_loss_weight=1.0,
):
    """
    Generic PINN training function.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network.

    x_collocation : torch.Tensor
        Collocation points.

    x0 : torch.Tensor
        Initial/boundary point.

    u0 : torch.Tensor
        Initial/boundary value.

    epochs : int
        Number of training epochs.

    learning_rate : float
        Learning rate for Adam.

    print_every : int
        Print training information every N epochs.

    residual_function : callable
        Function that computes the PDE/ODE residual.

    du0 : torch.Tensor or None
        Initial derivative value.

    derivative_loss_weight : float
        Weight of derivative initial condition.

    physics_loss_weight : float
        Weight of physics loss.

    initial_loss_weight : float
        Weight of value initial condition.

    Returns
    -------
    model
        Trained model.

    loss_history
        Total loss for every epoch.

    training_time
        Training duration.
    """

    if residual_function is None:
        residual_function = decay_residual

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    loss_history = []

    import time

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):

        optimizer.zero_grad()

        # ====================================================
        # Physics Loss
        # ====================================================

        residual = residual_function(
            model,
            x_collocation,
        )

        physics_loss = torch.mean(
            residual ** 2
        )

        # ====================================================
        # Initial Value Loss
        # ====================================================

        initial_loss = compute_initial_condition_loss(
            model,
            x0,
            u0,
        )

        # ====================================================
        # Initial Derivative Loss
        # ====================================================

        if du0 is not None:

            derivative_loss = compute_initial_derivative_loss(
                model,
                x0,
                du0,
            )

        else:

            derivative_loss = torch.tensor(
                0.0,
                dtype=physics_loss.dtype,
                device=physics_loss.device,
            )

        # ====================================================
        # Total Loss
        # ====================================================

        total_loss = (
            physics_loss_weight * physics_loss
            + initial_loss_weight * initial_loss
            + derivative_loss_weight * derivative_loss
        )

        # ====================================================
        # Backpropagation
        # ====================================================

        total_loss.backward()

        optimizer.step()

        # ====================================================
        # Store Loss
        # ====================================================

        loss_value = total_loss.detach().item()

        loss_history.append(
            loss_value
        )

        # ====================================================
        # Logging
        # ====================================================

        if (
            epoch == 1
            or epoch % print_every == 0
            or epoch == epochs
        ):

            print(
                f"Epoch {epoch:5d} | "
                f"Total Loss: {total_loss.item():.6e} | "
                f"Physics: {physics_loss.item():.6e} | "
                f"IC: {initial_loss.item():.6e} | "
                f"IC Derivative: "
                f"{derivative_loss.item():.6e}"
            )

    training_time = (
        time.perf_counter()
        - start_time
    )

    return (
        model,
        loss_history,
        training_time,
    )