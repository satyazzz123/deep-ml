import torch
import torch.nn.functional as F
import math
def sparse_window_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window_size: int, scale_factor: float = None) -> torch.Tensor:
    """Computes sparse attention with a sliding window mask.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: The radius of the attention window
        scale_factor: Scaling factor for the dot product. If None, uses sqrt(d_k).
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    Q=torch.as_tensor(Q,dtype=torch.float32)
    K=torch.as_tensor(K,dtype=torch.float32)
    V=torch.as_tensor(V,dtype=torch.float32)
    seq_len=Q.shape[0]
    
   
    w=window_size
    scores = torch.full((seq_len, seq_len), float("-inf"))

    for i in range(len(Q)):
      left = max(0, i - w)
      right = min(len(K), i + w + 1)
      scores[i,left:right]=(Q[i]@K[left:right,:].T)
    
    
    attention_score=F.softmax(scores/Q.shape[1]**0.5)
    return attention_score@V
    # return window