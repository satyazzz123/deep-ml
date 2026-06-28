import torch
import numpy as np

def deterministic_hash(s):
    '''Converts a string to a deterministic integer.'''
    h = 0
    for c in str(s):
        h = (h * 31 + ord(c)) % (2**31)
    return h

def create_hv(dim: int, seed: int) -> torch.Tensor:
    '''Creates a bipolar hypervector of given dimension using the seed.
    Returns a torch.Tensor of shape (dim,) with values in {-1, 1}.
    '''
    np.random.seed(seed % (2**32 - 1))
    hv = np.random.choice([-1, 1], dim)
    return torch.tensor(hv, dtype=torch.float32)

def create_row_hv(row: dict, dim: int, random_seeds: dict) -> torch.Tensor:
    '''Create composite hypervector for a dataset row using PyTorch operations.

    For each feature:
    1. Create a bipolar hypervector for the feature name
    2. Create a bipolar hypervector for the feature value
    3. Bind them via element-wise multiplication (torch.mul)

    Bundle all bound hypervectors via torch.sum, then normalize with torch.where.

    Hint: For each feature, the value seed should combine the base seed
    with the hashed value using modular arithmetic.

    Returns:
        torch.Tensor of shape (dim,) with bipolar values (-1 or 1).
    '''
    list_of_feature_seeds=[]
    for _,v in random_seeds.items():
      list_of_feature_seeds.append(v)
    list_of_value_seeds=[]

    for _,v in row.items():
      seed_value=deterministic_hash(v)
      list_of_value_seeds.append(seed_value)
    feature_vectors=[]
    value_vectors=[]
    for i in range(len(list_of_value_seeds)):
      feature_hv=create_hv(dim,list_of_feature_seeds[i])
      feature_vectors.append(feature_hv)
      value_hv=create_hv(dim,(list_of_value_seeds[i]+list_of_feature_seeds[i])%(2**31))
      value_vectors.append(value_hv)
    feature_vectors=torch.stack(feature_vectors)
    value_vectors=torch.stack(value_vectors)
    binded_vectors=torch.mul(feature_vectors,value_vectors)

    bundled_vector=torch.sum(binded_vectors,dim=0)


# If range is 0, keep original values (or set to 0.0), otherwise normalize
    normalized_x = torch.where(bundled_vector >= 0, 1, -1)

    return normalized_x



    