import torch
from typing import Tuple, Union

def adamax_optimizer(parameter: Union[float, torch.Tensor], 
                     grad: Union[float, torch.Tensor], 
                     m: Union[float, torch.Tensor], 
                     u: Union[float, torch.Tensor], 
                     t: int, 
                     learning_rate: float = 0.002, 
                     beta1: float = 0.9, 
                     beta2: float = 0.999, 
                     epsilon: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Update parameters using the Adamax optimizer.
    Adamax is a variant of Adam based on the infinity norm.
    It uses the maximum of past squared gradients instead of the exponential moving average.
    Args:
        parameter: Current parameter value
        grad: Current gradient
        m: First moment estimate
        u: Infinity norm estimate
        t: Current timestep
        learning_rate: Learning rate (default=0.002)
        beta1: First moment decay rate (default=0.9)
        beta2: Infinity norm decay rate (default=0.999)
        epsilon: Small constant for numerical stability (default=1e-8)
    Returns:
        tuple: (updated_parameter, updated_m, updated_u)
    """
    # Your code here
    parameter=torch.as_tensor(parameter,dtype=torch.float32)
    m=torch.as_tensor(m,dtype=torch.float32)
    u=torch.as_tensor(u,dtype=torch.float32)
    grad=torch.as_tensor(grad)
    grad_abs=torch.abs(grad)
    for time_step in range(t):
      m=beta1*m+(1-beta1)*grad
      u=torch.max(beta2*u,grad_abs)
      parameter-=learning_rate*(m/(1-beta1**t))*(1/(u+epsilon))
    return (parameter,m,u)
