from os import X_OK
import torch

def instance_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Instance Normalization over a 4D tensor X of shape (B, C, H, W).
    gamma: scale parameter of shape (C,)
    beta: shift parameter of shape (C,)
    epsilon: small value for numerical stability
    Returns: normalized tensor of same shape as X
    """
    X=torch.as_tensor(X,dtype=torch.float32)
    gamma=torch.as_tensor(gamma,dtype=torch.float32)
    beta=torch.as_tensor(beta,dtype=torch.float32)
    mean=torch.mean(X,dim=(-2,-1),keepdims=True)
    var=torch.var(X,dim=(-2,-1),keepdims=True,correction=0)
    X_normalized=(X-mean)/torch.sqrt((var+epsilon))

    return gamma*X_normalized+beta

    