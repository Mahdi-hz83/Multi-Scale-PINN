import torch


def residual(model, x):
    """
    Compute the physics residual for:

        u''(x) + u(x) = 0

    Parameters
    ----------
    model : torch.nn.Module
        PINN model.

    x : torch.Tensor
        Collocation points with shape [N, 1].

    Returns
    -------
    torch.Tensor
        Physics residual with shape [N, 1].
    """

    x = x.clone().detach().requires_grad_(True)

    u = model(x)

    # First derivative du/dx
    du_dx = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Second derivative d2u/dx2
    d2u_dx2 = torch.autograd.grad(
        outputs=du_dx,
        inputs=x,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Physics residual:
    #
    # u'' + u = 0
    residual_value = d2u_dx2 + u

    return residual_value


def exact_solution(x):
    """
    Exact solution:

        u(x) = sin(x)
    """

    return torch.sin(x)