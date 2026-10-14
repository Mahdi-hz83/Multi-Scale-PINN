import os
import sys

import torch

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.pinn import PINN # type: ignore


def main():

    # --------------------------------------------------
    # 1. Create model
    # --------------------------------------------------

    model = PINN(
        input_dim=1,
        hidden_dim=32,
        hidden_layers=3,
        output_dim=1,
    )

    # --------------------------------------------------
    # 2. Create input
    # --------------------------------------------------

    x = torch.tensor(
        [
            [0.0],
            [0.25],
            [0.5],
            [0.75],
            [1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    # --------------------------------------------------
    # 3. Forward pass
    # --------------------------------------------------

    u = model(x)

    # --------------------------------------------------
    # 4. Compute du/dx
    # --------------------------------------------------

    du_dx = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    # --------------------------------------------------
    # 5. Print results
    # --------------------------------------------------

    print("=" * 60)
    print("Automatic Differentiation Test")
    print("=" * 60)

    print("\nInput x:")
    print(x)

    print("\nNetwork output u(x):")
    print(u)

    print("\nDerivative du/dx:")
    print(du_dx)

    print("\nShapes:")
    print("x     :", x.shape)
    print("u     :", u.shape)
    print("du/dx :", du_dx.shape)

    print("=" * 60)


if __name__ == "__main__":
    main()