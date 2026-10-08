import torch

def rmsprop_update(
    params: torch.Tensor,
    grads: torch.Tensor,
    cache: torch.Tensor,
    lr: float = 0.01,
    beta: float = 0.9,
    epsilon: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform RMSProp optimization update.
    """

    # 1. Update moving average of squared gradients
    updated_cache = beta * cache + (1 - beta) * (grads ** 2)

    # 2. Update parameters
    updated_params = params - lr * grads / (torch.sqrt(updated_cache) + epsilon)

    return updated_params, updated_cache