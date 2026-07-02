import torch

def calculate_dot_product(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the dot product of two vectors.
    Args:
        vec1 (torch.Tensor): 1D tensor representing the first vector.
        vec2 (torch.Tensor): 1D tensor representing the second vector.
    Returns:
        torch.Tensor: The dot product of the two vectors as a scalar tensor.
    """
    # Your code here
    vec1=torch.as_tensor(vec1,dtype=torch.float32)
    vec2=torch.as_tensor(vec2,dtype=torch.float32)
    dot=torch.sum(vec1*vec2)
    return dot
    
    

    