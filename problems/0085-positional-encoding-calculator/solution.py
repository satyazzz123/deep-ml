import math
import torch
import numpy as np


def pos_encoding(position: int, d_model: int):
    """
    Compute positional encodings for Transformer models.

    Args:
        position: sequence length (number of positions)
        d_model: model dimensionality

    Returns:
        torch.Tensor of shape (position, d_model) with dtype float16,
        or -1 if position == 0 or d_model <= 0.
    """
    # Your code here
    position_encodings=[]
    if position<=0 or d_model <=0:
      return -1
    for p in range(position):
      temp=[]
      for d in range(d_model):
        if d%2==0:
          temp.append(math.sin(p/(10000**(d/d_model))))
        else:
          temp.append(math.cos(p/(10000**((d-1)/d_model))))
      position_encodings.append(temp)
    position_encodings=torch.tensor(position_encodings,dtype=torch.float16)

    return position_encodings
