import torch
import torch.nn.functional as F
def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    X=torch.as_tensor(X,dtype=torch.float32)
    W_q=torch.as_tensor(W_q,dtype=torch.float32)
    W_k=torch.as_tensor(W_k,dtype=torch.float32)
    W_v=torch.as_tensor(W_v,dtype=torch.float32)
    

    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    Q=torch.as_tensor(Q,dtype=torch.float32)
    K=torch.as_tensor(K,dtype=torch.float32)
    V=torch.as_tensor(V,dtype=torch.float32)
    seq_len,d_k=Q.shape
    attention_scores=Q@K.T
    # attention_scores=torch.as_tensor(attention_scores,dtype=torch.float32)
    attention_scores=attention_scores/d_k**(0.5)
    # attention_mask=torch.triu(torch.ones_like(attention_scores),diagonal=1).bool()
    # attention_scores=attention_scores.masked_fill(attention_mask,float('-inf'))
    attention_weights=F.softmax(attention_scores,dim=1)
    return attention_weights@V
    # return attention_scores

    
