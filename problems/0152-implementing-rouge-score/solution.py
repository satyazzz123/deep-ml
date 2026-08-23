import torch
from collections import Counter

def rouge_1_score(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1 score between reference and candidate texts.
    
    Returns a dictionary with precision, recall, and f1 as torch tensors.
    """
    # Your code here
    candidate_count=Counter(candidate.split(" "))
    reference_count=Counter(reference.split(" "))
    overlap=0
    for k,v in dict(candidate_count).items():
      if k in dict(reference_count):
        # print(reference_count[k])
        overlap+=min(v,reference_count[k])
        # print(overlap)
    precision=overlap/(len(candidate.split(" ")))
    recall=overlap/(len(reference.split(" ")))
    if precision==0:
      f1_score=0
    else:
      f1_score=(2*precision*recall)/(precision+recall)

    

    

    return {
        "precision": torch.tensor(precision, dtype=torch.float64),
        "recall": torch.tensor(recall, dtype=torch.float64),
        "f1": torch.tensor(f1_score, dtype=torch.float64)
    }