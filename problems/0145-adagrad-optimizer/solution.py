import torch

def adagrad_optimizer(parameter: torch.Tensor, grad: torch.Tensor, G: torch.Tensor, learning_rate: float = 0.01, epsilon: float = 1e-8) -> tuple:
    """
    Update parameters using the Adagrad optimizer.
    Adapts the learning rate for each parameter based on the historical gradients.
    Args:
        parameter: Current parameter value (torch.Tensor)
        grad: Current gradient (torch.Tensor)
        G: Accumulated squared gradients (torch.Tensor)
        learning_rate: Learning rate (default=0.01)
        epsilon: Small constant for numerical stability (default=1e-8)
    Returns:
        tuple: (updated_parameter, updated_G)
    """
    parameter=torch.as_tensor(parameter,dtype=torch.float32)
    grad=torch.as_tensor(grad,dtype=torch.float32)
    G=torch.as_tensor(G,dtype=torch.float32)

    G+=grad**2
    parameter-=(learning_rate/((G**0.5)+epsilon ))*grad
    return torch.round(parameter, decimals=5), torch.round(G, decimals=5)