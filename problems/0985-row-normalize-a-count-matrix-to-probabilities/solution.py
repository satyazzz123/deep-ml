import numpy as np

def row_normalize(counts: list[list[float]]) -> list[list[float]]:
    """Convert a count matrix into a row-stochastic probability matrix."""
    counts=np.array(counts)
    val=counts/(np.sum(counts,axis=-1,keepdims=True)+1e-10)
    return val