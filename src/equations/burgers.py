"""Viscous 1D Burgers equation utilities for PINN experiments.

Equation
--------
    u_t + u u_x - nu u_xx = 0,
    x in [-1, 1], t in [0, 1],
    u(x, 0) = -sin(pi x),
    u(-1, t) = u(1, t) = 0.

The default viscosity is nu = 0.01 / pi, matching the standard Burgers
benchmark commonly used for PINN evaluation.
"""

import math
import torch

DEFAULT_NU = 0.01 / math.pi


def initial_condition(x):
    return -torch.sin(math.pi * x)


def residual(model, xt, nu=DEFAULT_NU):
    """Compute Burgers PDE residual using PyTorch automatic differentiation."""
    xt = xt.clone().detach().requires_grad_(True)
    u = model(xt)

    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=xt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    grad_u_x = torch.autograd.grad(
        outputs=u_x,
        inputs=xt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_xx = grad_u_x[:, 0:1]

    return u_t + u * u_x - nu * u_xx


def derivatives(model, xt):
    """Return u, u_x, u_t and u_xx for validation/testing."""
    xt = xt.clone().detach().requires_grad_(True)
    u = model(xt)
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=xt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]
    grad_u_x = torch.autograd.grad(
        outputs=u_x,
        inputs=xt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_xx = grad_u_x[:, 0:1]
    return u, u_x, u_t, u_xx
