import torch
from collections import Counter

def probability(total_sample:int, number_of_sample:int)->int:
    return number_of_sample/total_sample


def disorder(apples: torch.Tensor) -> torch.Tensor:
    """
    Compute the disorder in a basket of apples.
    """
    # Your code here
    apples=torch.as_tensor(apples,dtype=torch.float32)
    count=Counter(apples.tolist())
    apple_dict=dict(count)
    total_sample=len(apples)
    probs=0
    for _,sample in apple_dict.items():
      # print(k ,"--",v)
      temp=probability(total_sample,sample)
      probs+=temp**2




    
    return torch.tensor(1-probs)


