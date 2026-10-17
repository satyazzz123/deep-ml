import numpy as np

def softmax(q):
    # Subtract max for numerical stability before exponentiating
    exp_q = np.exp(q - np.max(q, axis=-1, keepdims=True))
    return exp_q / np.sum(exp_q, axis=-1, keepdims=True)

def transfer_and_attend(X, W_q_lin, W_k_lin, W_v_lin):
    """
    Transfer linear-style weights (shape (d_out, d_in)) into parameter-style
    weights (shape (d_in, d_out)) and compute single-head causal self-attention.

    Args:
        X: input array of shape (seq_len, d_in)
        W_q_lin, W_k_lin, W_v_lin: linear-style weight matrices of shape (d_out, d_in)

    Returns:
        Nested list of shape (seq_len, d_out): the self-attention output.
    """
    # Transpose linear weights to parameter-style (d_in, d_out)
    W_q = np.asarray(W_q_lin).T
    W_k = np.asarray(W_k_lin).T
    W_v = np.asarray(W_v_lin).T
    X=np.asarray(X)
    d_k = W_k.shape[1]
    seq_len = X.shape[0]

    # Compute Query, Key, and Value matrices
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    # Compute scaled dot-product attention scores
    scores = (Q @ K.T) / (d_k ** 0.5)

    # Apply causal mask: upper triangle above the main diagonal set to -inf
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    scores = scores + mask

    # Compute attention probabilities and final output
    attention_probs = softmax(scores)
    self_attention = attention_probs @ V

    # Return as a nested list to match docstring specification
    return self_attention.tolist()