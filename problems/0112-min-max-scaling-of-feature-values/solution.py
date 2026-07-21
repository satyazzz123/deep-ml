import torch

def min_max(x: torch.Tensor) -> torch.Tensor:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A tensor of numerical values
    
    Returns:
        A new tensor with values normalized to [0, 1]
    """
    # Your code here
    x=torch.as_tensor(x,dtype=torch.float32)
    x_max,_=torch.max(x,dim=-1)
    x_min,_=torch.min(x,dim=-1)
    
    
    return (x-x_min)/(x_max-x_min)