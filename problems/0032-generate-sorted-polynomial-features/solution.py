import torch
from itertools import combinations_with_replacement
import math
def polynomial_features(X, degree):
    """
    Given a 2D tensor X and integer degree, return a new tensor of all polynomial feature combinations
    (with constant term), sorted for each sample from smallest to largest.
    """
    X=X.tolist()
    temp=[]
    for i in X:
      i.insert(0,1)
      z=list(combinations_with_replacement(i, degree))
      temp.append(z)
    ans=[]
    for i in range(len(temp)):
      for j in range(len(temp[0])):
        val=math.prod(temp[i][j]) 
        temp[i][j]=val
    return torch.tensor(temp,dtype=torch.float32)

    
