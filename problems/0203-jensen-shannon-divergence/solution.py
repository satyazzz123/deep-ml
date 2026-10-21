import torch

def jensen_shannon_divergence(P: torch.Tensor, Q: torch.Tensor) -> float:
    """
    Compute the Jensen-Shannon Divergence between two probability distributions.
    
    Args:
        P: First probability distribution
        Q: Second probability distribution
    
    Returns:
        Jensen-Shannon Divergence value
    """
    # Your code here
    P=torch.as_tensor(P,dtype=torch.float32)
    Q=torch.as_tensor(Q,dtype=torch.float32)
    P=P+1e-7
    Q=Q+1e-7
    M=(P+Q)/2

    D_kl_P=torch.dot(P,torch.log(P/(M)))
    D_kl_Q=torch.dot(Q,torch.log(Q/(M)))
    return ((D_kl_P+D_kl_Q)/2).item()
