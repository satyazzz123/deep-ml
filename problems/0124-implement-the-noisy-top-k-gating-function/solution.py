import torch
import torch.nn.functional as F
def noisy_topk_gating(
    X: torch.Tensor,
    W_g: torch.Tensor,
    W_noise: torch.Tensor,
    N: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    # Your code here
    X=torch.as_tensor(X,dtype=torch.float32)
    W_g=torch.as_tensor(W_g,dtype=torch.float32)
    W_noise=torch.as_tensor(W_noise,dtype=torch.float32)
    N=torch.as_tensor(N,dtype=torch.float32)

    H_base=X@W_g
    H_noise=X@W_noise
    H=H_base+N*torch.log(torch.exp(H_noise)+1)

    _, indices = torch.topk(H,k, dim=-1)
    z = torch.full(H.shape, float('-inf'))

    z.scatter_(dim=-1, index=indices, value=float(0))
    H=H+z
    probs_func = F.softmax(H, dim=-1)
    return probs_func




