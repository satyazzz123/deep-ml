import torch

def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Layer Normalization.
    """
    # Your code here
    x=torch.as_tensor(X,dtype=torch.float32)
    variance, mean = torch.var_mean(x, dim=-1,keepdim=True,unbiased=False)
    gamma=torch.tensor(gamma,dtype=torch.float32)
    beta=torch.tensor(beta,dtype=torch.float32)
    # mean=mean.reshape(-1,x.shape[1],1)
    # variance=variance.reshape(-1,x.shape[1],1)


    normalized=(x-mean)/(variance+epsilon)**0.5


    return gamma*normalized+beta