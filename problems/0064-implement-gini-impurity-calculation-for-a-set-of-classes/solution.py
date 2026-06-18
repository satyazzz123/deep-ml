import torch
from collections import Counter
def gini_impurity(y: torch.Tensor) -> float:
    """
    Calculate Gini Impurity for a tensor of class labels.

    :param y: 1D Tensor of class labels (integer type)
    :return: Gini Impurity rounded to three decimal places
    """
    y=torch.as_tensor(y,dtype=torch.float32)
    y=y.tolist()
    x=Counter(y)
    probs_list=[]
    for k,v in x.items():
      probs_list.append(v)
    probs_list=torch.tensor(probs_list,dtype=torch.float32)
    probs_list/=len(y)
    gini=torch.sum(torch.square(probs_list))
    return (1-gini).item()







    