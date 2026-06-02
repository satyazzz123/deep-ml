import torch

def relu(z: float) -> torch.Tensor:
    """
    Implements the ReLU activation function using PyTorch.
    
    Args:
        z: A float input value.
    
    Returns:
        A torch.Tensor with ReLU applied (max(0, z)).
    """
    # Your code here
    z=torch.as_tensor(z,dtype=torch.float32)
    if z.item()<0:
        return torch.tensor(0)
    else:
        return z
