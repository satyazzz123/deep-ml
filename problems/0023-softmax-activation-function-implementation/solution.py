import torch
import torch.nn.functional as F
import math
def exp(x):
  return math.exp(x)

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here
    
    scores=torch.as_tensor(scores,dtype=torch.float32)
    exp_scores=scores.apply_(exp)
    denominator=exp_scores.sum()
    probs=(exp_scores/denominator)
    probs=torch.round(probs,decimals=4)
    return [round(p, 4) for p in probs.tolist()]

    
    


