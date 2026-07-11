import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    X=torch.as_tensor(X,dtype=torch.float32)
    W_q=torch.as_tensor(W_q,dtype=torch.float32)
    W_k=torch.as_tensor(W_k,dtype=torch.float32)
    W_v=torch.as_tensor(W_v,dtype=torch.float32)
    return (X@W_q,X@W_k,X@W_v)



def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    _,d_k=Q.shape
    attention_score=(Q@K.T)/d_k**0.5
    attention_score=F.softmax(attention_score,dim=-1)
    attention_weights=attention_score@V
    return attention_weights

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len,d_model=Q.shape
    Q=Q.reshape(seq_len,n_heads,-1)
    K=K.reshape(seq_len,n_heads,-1)
    V=V.reshape(seq_len,n_heads,-1)
    Q=Q.permute(1,0,2)
    K=K.permute(1,0,2)
    V=V.permute(1,0,2)
    
    attention_list=[]
    for i in range(n_heads):
      attention_weights=self_attention(Q[i],K[i],V[i])
      attention_list.append(attention_weights)
    final_tensor = torch.cat(attention_list, dim=-1)
    return final_tensor





