import torch
import torch.nn.functional as F

def cross_entropy_derivative(logits: torch.Tensor, target: int) -> torch.Tensor:
    """
    Compute the derivative of cross-entropy loss with respect to logits.
    
    Args:
        logits: Raw model outputs tensor
        target: Index of the true class
        
    Returns:
        Gradient tensor
    """
    # Your code here - can use autograd or the analytical formula
    logits=torch.as_tensor(logits,dtype=torch.float32)
    softmax_values=F.softmax(logits)
    ohe=torch.zeros_like(logits)
    ohe[target]=torch.tensor(1,dtype=torch.float32)
    return (softmax_values-ohe)