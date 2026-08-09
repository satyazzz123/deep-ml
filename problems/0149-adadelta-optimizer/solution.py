import torch
from typing import Tuple

def adadelta_optimizer(
    parameter: torch.Tensor,
    grad: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    rho: float = 0.95,
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    parameter = torch.as_tensor(parameter, dtype=torch.float32)
    grad = torch.as_tensor(grad, dtype=torch.float32)
    u = torch.as_tensor(u, dtype=torch.float32)
    v = torch.as_tensor(v, dtype=torch.float32)

    # Running average of squared gradients
    u = rho * u + (1 - rho) * grad**2

    # RMS of gradients
    RMS_grad = torch.sqrt(u + epsilon)

    # RMS of previous parameter updates
    RMS_update = torch.sqrt(v + epsilon)

    # New parameter update
    delta = -grad * (RMS_update / RMS_grad)

    # Update parameter
    parameter += delta

    # Running average of squared parameter updates
    v = rho * v + (1 - rho) * delta**2

    return parameter, u, v