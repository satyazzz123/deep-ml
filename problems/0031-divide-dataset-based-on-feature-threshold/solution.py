import torch
from typing import List

def divide_on_feature(X, feature_i, threshold) -> List[torch.Tensor]:
    """
    Divide the tensor X into two subsets based on whether X[:, feature_i] is greater than or equal to (or equal to) the threshold.
    Return two tensors: one with samples that meet the condition, one with samples that do not.
    """
    accepted=[]
    rejected=[]
    for i in X:
      if i[feature_i].item()>=threshold:
        accepted.append(i)
      else:
        rejected.append(i)
    
    accepted=torch.tensor(accepted)
    rejected=torch.tensor(rejected)
    return[accepted,rejected]

        
