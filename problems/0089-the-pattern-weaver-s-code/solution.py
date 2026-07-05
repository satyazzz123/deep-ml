from os import W_OK
import torch
import torch.nn.functional as F

def pattern_weaver(n: int, crystal_values: list, dimension: int) -> torch.Tensor:
    """
    Implements a simplified self-attention mechanism for crystal values.
    
    Args:
        n: Number of crystals
        crystal_values: List of crystal values
        dimension: Scaling dimension for attention scores
    
    Returns:
        torch.Tensor of final weighted patterns for each crystal
    """
    # Convert inputs to tensor
    values = torch.tensor(crystal_values, dtype=torch.float64).reshape(-1, 1)

    Q = values
    K = values
    V = values

    scores = (Q @ K.T) / (dimension ** 0.5)
    weights = F.softmax(scores, dim=-1)
    output = weights @ V

    return output.squeeze()
    
  