import torch
import math
def selu(x: torch.Tensor) -> torch.Tensor:
    """
    Implements the SELU (Scaled Exponential Linear Unit) activation function.

    Args:
        x: Input tensor

    Returns:
        SELU activation tensor
    """
    x=torch.as_tensor(x,dtype=torch.float32)
    alpha = 1.6732632423543772
    scale = 1.0507009873554804
    return torch.where(x>0,scale*x,scale*(alpha*(torch.exp(x)-1)))
    
    