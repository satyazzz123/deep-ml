import torch

def group_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Group Normalization on a 4D input tensor.
    
    Args:
        X: torch tensor of shape (B, C, H, W), input data
        gamma: torch tensor of shape (1, C, 1, 1), scale parameter
        beta: torch tensor of shape (1, C, 1, 1), shift parameter
        num_groups: number of groups for normalization
        epsilon: small constant to avoid division by zero
    
    Returns:
        norm_X: torch tensor of shape (B, C, H, W), normalized output
    """
    X=torch.as_tensor(X,dtype=torch.float32)
    gamma=torch.as_tensor(gamma,dtype=torch.float32)
    beta=torch.as_tensor(beta,dtype=torch.float32)
    B,C,H,W=X.shape
    group_size=C//num_groups
    X=X.reshape(B,num_groups,group_size,H,W)
    mean=torch.mean(X,dim=(2,3,4),keepdim=True)
    variance=torch.var(X,dim=(2,3,4),keepdim=True,correction=0)
    normalization=(X-mean)/torch.sqrt((variance+epsilon))
    print(normalization.shape)
    y_hat=gamma*normalization.reshape(B,num_groups*group_size,H,W)+beta
    
    return y_hat
