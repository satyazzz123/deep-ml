import numpy as np

def embedding_via_one_hot(token_ids, W):
    """
    Compute token embeddings via one-hot encoding and matrix multiplication.

    Args:
        token_ids: list or 1D array of integer token IDs
        W: numpy array of shape (vocab_size, embed_dim)

    Returns:
        numpy array of shape (len(token_ids), embed_dim)
    """
    token_ids=np.array(token_ids)
    W=np.array(W)
    token_ohe=np.zeros((len(token_ids),W.shape[0]))

    for i in range(len(token_ohe)):
      token_ohe[i,token_ids[i]]=1

    return token_ohe@W