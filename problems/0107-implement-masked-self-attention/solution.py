import torch
import torch.nn.functional as F
def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    X=torch.as_tensor(X,dtype=torch.float32)
    W_q=torch.as_tensor(W_q,dtype=torch.float32)
    W_k=torch.as_tensor(W_k,dtype=torch.float32)
    W_v=torch.as_tensor(W_v,dtype=torch.float32)
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked self-attention.
    """

    _,d_k=Q.shape
    mask=torch.as_tensor(mask,dtype=torch.float32)
    attention_score=(Q@K.T)/(d_k**0.5)
    attention_score=attention_score+mask
    attention_score=F.softmax(attention_score,dim=-1)
    attention_weight=attention_score@V
    # return mask
    return attention_weight

   
    