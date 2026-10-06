import torch

def clip_gradients_by_global_norm(
    gradients: list[torch.Tensor],
    max_norm: float
) -> list[torch.Tensor]:

    # 1. Compute ONE global norm across all gradients
    global_norm = torch.sqrt(
        sum(torch.sum(grad ** 2) for grad in gradients)
    )

    # 2. Compute ONE clipping coefficient
    clipping_coefficient = max_norm / (global_norm + 1e-10)

    # 3. Only clip if global norm exceeds max_norm
    if clipping_coefficient < 1:
        return [
            clipping_coefficient * grad
            for grad in gradients
        ]

    return gradients