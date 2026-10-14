import torch


def exact_solution(x):
    """
    Exact solution of:

        u'(x) + u(x) = 0
        u(0) = 1

    Exact solution:

        u(x) = exp(-x)
    """
    return torch.exp(-x)


def residual(model, x):
    """
    Physics residual of:

        u'(x) + u(x) = 0

    Therefore:

        r(x) = du/dx + u
    """

    # Clone x so that we do not modify the original tensor.
    x = x.clone().detach().requires_grad_(True)

    # Neural network prediction.
    u = model(x)

    # Compute du/dx using PyTorch Automatic Differentiation.
    du_dx = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    # ODE residual.
    residual_value = du_dx + u

    return residual_value