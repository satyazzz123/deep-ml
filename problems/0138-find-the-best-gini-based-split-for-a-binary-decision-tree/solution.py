import torch
from typing import Tuple
from collections import Counter
def gini_impurity(y:torch.Tensor):
  counter=Counter(y.tolist())
  probs=[]
  for k,v in dict(counter).items():
    probs.append(v/len(y))

  return 1-torch.sum(torch.tensor(probs)**2)

def find_best_split(X: torch.Tensor, y: torch.Tensor) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    # ✏️ TODO: implement
    X=torch.as_tensor(X,dtype=torch.float32)
    y=torch.as_tensor(y,dtype=torch.float32)
    final_G=10000
    feature_idx=0
    element_idx=0
    for i in range(X.shape[-1]):
      feature=X[:,i]
      for j in range(len(feature)):
        mask=(feature<=feature[j])
        reverse_mask=(feature>feature[j])
        left_split=gini_impurity(y[mask])
        right_split=gini_impurity(y[reverse_mask])
        G_split=(len(y[mask])/len(y))*left_split + (len(y[reverse_mask])/len(y))*right_split
        if G_split<final_G:
          feature_idx=i
          final_G=G_split
          element_idx=j

    threshold = X[element_idx, feature_idx].item()

    if threshold.is_integer():
      threshold = int(threshold)

    return feature_idx, threshold
    