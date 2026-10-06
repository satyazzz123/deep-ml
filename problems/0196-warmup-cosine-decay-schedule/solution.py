import torch
import math

def warmup_cosine_schedule(T: int, W: int, lr_max: float, lr_min: float) -> torch.Tensor:
    """
    Compute learning rate schedule with linear warmup and cosine decay.
    
    Args:
        T: Total number of training steps
        W: Number of warmup steps
        lr_max: Maximum learning rate (reached after warmup)
        lr_min: Minimum learning rate (reached at end of training)
    
    Returns:
        Tensor of learning rates for each step
    """
    # Your code here
    lr_rate=[]
    for t in range(W):
      n_t=(t/W)*lr_max
      lr_rate.append(n_t)
    
    for t in range(W,T):
      n_t=lr_min+0.5*(lr_max-lr_min)*(1+math.cos(((t-W)*math.pi)/(T-W)))
      lr_rate.append(n_t)
    return torch.tensor(lr_rate)

    
