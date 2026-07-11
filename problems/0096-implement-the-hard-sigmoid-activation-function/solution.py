import torch

def hard_sigmoid(x: float) -> float:
    """
    Implements the Hard Sigmoid activation function using PyTorch.
    Uses the Keras convention: 0.2*x + 0.5, clamped to [0, 1].

    Args:
        x (float): Input value

    Returns:
        float: The Hard Sigmoid of the input
    """
    # Your code here
    if x<=-1*2.5:
      return 0
    if x>=2.5:
      return 1
    return 0.2*x+0.5

