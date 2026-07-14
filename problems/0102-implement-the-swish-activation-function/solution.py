import torch

def swish(x: torch.Tensor) -> torch.Tensor:
    """
    Implements the Swish activation function.

    Args:
        x: Input tensor

    Returns:
        The Swish activation value as a tensor
    """
    # Your code here
    x=torch.as_tensor(x,dtype=torch.float32)
    exp_part=torch.exp(-1*x)
    return x*(1/(1+exp_part))
