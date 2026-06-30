import torch
import torch.nn.functional as F

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Calculate the cosine similarity of two vectors using PyTorch.
    Args:
        v1 (torch.Tensor): 1D tensor representing the first vector.
        v2 (torch.Tensor): 1D tensor representing the second vector.
    Returns:
        float: The cosine similarity of the two vectors.
    """
    # Implement your code here
    v1=torch.as_tensor(v1,dtype=torch.float32)
    v2=torch.as_tensor(v2,dtype=torch.float32)

    return round((torch.dot(v1,v2).item()/(torch.sum(torch.square(v1)).item()*torch.sum(torch.square(v2)).item())**0.5),3)