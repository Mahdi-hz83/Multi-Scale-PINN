import torch


def physics_loss(residual_value):
    """
    Mean Squared Error of the physics residual.
    """

    return torch.mean(
        residual_value ** 2
    )


def initial_condition_loss(
    model,
    x0,
    u0,
):
    """
    Mean Squared Error of the initial condition.
    """

    prediction = model(x0)

    return torch.mean(
        (prediction - u0) ** 2
    )


def total_loss(
    model,
    residual_value,
    x0,
    u0,
):
    """
    Total PINN loss:

        L_total = L_physics + L_initial
    """

    loss_physics = physics_loss(
        residual_value
    )

    loss_initial = initial_condition_loss(
        model,
        x0,
        u0,
    )

    loss = (
        loss_physics
        + loss_initial
    )

    return (
        loss,
        loss_physics,
        loss_initial,
    )