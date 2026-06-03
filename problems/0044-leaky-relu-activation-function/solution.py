import torch

def leaky_relu(z: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """
    Implements the Leaky ReLU activation function using PyTorch.
    
    Args:
        z: Input tensor (scalar or any shape)
        alpha: Slope for negative values (default: 0.01)
    
    Returns:
        Output tensor after applying Leaky ReLU
    """
    # Your implementation here
    z=torch.as_tensor(z,dtype=torch.float32)
    if z.item()>=0:
        return z 
    else:
        return alpha*z

