import torch


def mse(prediction, target):
    """
    Mean Squared Error.
    """

    return torch.mean(
        (prediction - target) ** 2
    ).item()


def mae(prediction, target):
    """
    Mean Absolute Error.
    """

    return torch.mean(
        torch.abs(prediction - target)
    ).item()


def relative_l2_error(prediction, target):
    """
    Relative L2 Error:

        ||prediction - target||_2
        -------------------------
              ||target||_2
    """

    numerator = torch.linalg.norm(
        prediction - target
    )

    denominator = torch.linalg.norm(
        target
    )

    return (
        numerator / denominator
    ).item()


def max_absolute_error(prediction, target):
    """
    Maximum Absolute Error.
    """

    return torch.max(
        torch.abs(
            prediction - target
        )
    ).item()