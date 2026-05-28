import numpy as np
from typing import List, Tuple

def get_random_indices(X, replacements=True) -> list:
    """
    Generate n_subsets random subsets from the dataset (X, y).
    Each subset is a tuple (X_subset, y_subset), where both are lists.
    
    Args:
        X: 2D array of shape (n_samples, n_features)
        y: 1D array of shape (n_samples,)
        n_subsets: Number of subsets to generate
        replacements: If True, sample with replacement
                      If False, sample without replacement
    
    Returns:
        List of (X_subset, y_subset) tuples
    """
    # Your code here
    # X=np.array(X)
    x_size=len(X)
    indexes=[]
    if replacements:
      for _ in range(x_size):
        num = np.random.randint(low=0, high=x_size)
        indexes.append(num)
    else:
      w=np.arange(x_size)
      np.random.shuffle(w)
      indexes=w[:x_size//2]
      

    



    return indexes

def get_random_subsets(X, y, n_subsets, replacements=True) -> list:
    """
    Generate n_subsets random subsets from the dataset (X, y).
    Each subset is a tuple (X_subset, y_subset), where both are lists.
    
    Args:
        X: 2D array of shape (n_samples, n_features)
        y: 1D array of shape (n_samples,)
        n_subsets: Number of subsets to generate
        replacements: If True, sample with replacement
                      If False, sample without replacement
    
    Returns:
        List of (X_subset, y_subset) tuples
    """
    # Your code here

    subset_final_list=[]
    for i in range(n_subsets):
      idx=get_random_indices(X,replacements)

      subset_final=(X[idx].tolist(),y[idx].tolist())
      subset_final_list.append(subset_final)
    return subset_final_list







    
    



