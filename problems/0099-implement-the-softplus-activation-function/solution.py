import torch
import torch.nn.functional as F
import math
def softplus(x: float) -> torch.Tensor:
    """
    Compute the softplus activation function using PyTorch.

    Args:
        x: Input value

    Returns:
        The softplus value: log(1 + e^x) as a torch.Tensor
    """
    x=math.log(1+math.exp(x))
    return torch.tensor(x)
    