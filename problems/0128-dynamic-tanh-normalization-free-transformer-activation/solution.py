import torch
import torch.nn.functional as F
def dynamic_tanh(x: torch.Tensor, alpha: float, gamma: torch.Tensor, beta: torch.Tensor) -> list:
    """
    Implement the Dynamic Tanh (DyT) function.
    
    Args:
        x: Input tensor
        alpha: Scaling factor for input
        gamma: Output scaling factor
        beta: Output shift factor
    
    Returns:
        List of transformed values
    """
    x=torch.as_tensor(x,dtype=torch.float32)
    gamma=torch.as_tensor(gamma,dtype=torch.float32)
    beta=torch.tensor(beta,dtype=torch.float32)

    out=F.tanh(alpha*x)
    print(out)
    return (gamma*out+beta).tolist()