import torch
from collections import Counter
def empirical_pmf(samples: torch.Tensor) -> list:
    """
    Given a 1D tensor of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    samples=torch.as_tensor(samples)
    counter=Counter(samples.tolist())
    emp=[]
    for k,v in dict(counter).items():
      emp.append((k,v/len(samples)))

    return emp
