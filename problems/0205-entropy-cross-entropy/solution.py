import torch

def entropy_and_cross_entropy(P: torch.Tensor, Q: torch.Tensor) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.
    
    Args:
        P: True probability distribution (torch.Tensor)
        Q: Predicted probability distribution (torch.Tensor)
    
    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    # Your code here
    P=torch.as_tensor(P)
    Q=torch.as_tensor(Q)

    entropy=-1*torch.sum(torch.log(P+1e-10)*P)
    cross_entropy=-1*torch.dot(torch.log(Q+1e-10),P)
    return (entropy.item(),cross_entropy.item())