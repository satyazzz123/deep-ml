import torch

def conditional_probability(data: torch.Tensor, x: int, y: int) -> float:
    """
    Returns the probability P(Y=y|X=x) from tensor of (X, Y) pairs.
    Args:
      data: Tensor of shape (N, 2) containing encoded (X, Y) pairs
      x: value of X to condition on (encoded as integer)
      y: value of Y to check (encoded as integer)
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    # Your code here
    data=torch.as_tensor(data,dtype=torch.float32)
    data=data.T
    X=data[0]
    Y=data[1]
    X_mask=(X==x)
    
    XY_mask=(X==x) & (Y==y)
    return (torch.sum(XY_mask)/(torch.sum(X_mask)+1e-10)).item()
