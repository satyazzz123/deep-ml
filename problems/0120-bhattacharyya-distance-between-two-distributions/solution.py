import torch

def bhattacharyya_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    Calculate the Bhattacharyya distance between two probability distributions.
    
    Args:
        p: First probability distribution as a 1D tensor
        q: Second probability distribution as a 1D tensor
    
    Returns:
        The Bhattacharyya distance rounded to 4 decimal places.
        Returns 0.0 if inputs have different lengths or are empty.
    """
    p=torch.as_tensor(p,dtype=torch.float32)
    q=torch.as_tensor(q,dtype=torch.float32)
    if len(p)==0 or len(q)==0 or len(q)!=len(p):
      return 0
    bc=torch.sum((torch.sqrt(p*q)),dim=-1)
    bd=-torch.log(bc)
    return round(bd.item(),4)

