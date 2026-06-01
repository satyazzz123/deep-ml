import torch
from typing import List
import math
def exp(x):
  return math.exp(x)
def log(x):
  return math.log(x)
def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    scores=torch.as_tensor(scores,dtype=torch.float32)
    scores=scores.apply_(exp)
    scores=scores/torch.sum(scores,dim=-1)
    
    return scores.apply_(log)

