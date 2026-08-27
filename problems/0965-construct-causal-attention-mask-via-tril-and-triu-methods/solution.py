import numpy as np
import math

def softmax(matrix):
  e_x=np.exp(matrix - np.max(matrix, axis=-1, keepdims=True))
  e_x_sum=np.sum(e_x,axis=-1,keepdims=True)
  return e_x/e_x_sum
def causal_mask_attention(attn_weights: np.ndarray, method: str = 'tril') -> list:
    """Apply causal masking two ways and return the resulting attention matrix as a nested list."""
    if method =="tril":
      mask=np.tril(np.ones(attn_weights.shape,dtype=bool))
      mask=np.where(mask,1,0)
      attn_weights=attn_weights*mask
      attn_weights=attn_weights/np.sum(attn_weights,axis=-1,keepdims=True)
    if method=="triu":
      attn_weights=np.log(attn_weights)
      mask=np.triu(np.ones(attn_weights.shape,dtype=bool),k=1)
      attn_weights[mask]=-math.inf
      attn_weights=softmax(attn_weights)
    
    
    
    
    

    return attn_weights

