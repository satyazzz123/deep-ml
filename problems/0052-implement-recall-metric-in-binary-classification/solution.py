import torch
from collections import Counter
def recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Tensor of true binary labels (0 or 1)
        y_pred: Tensor of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    # Your code here
    y_true=torch.as_tensor(y_true,dtype=torch.float32)
    y_pred=torch.as_tensor(y_pred,dtype=torch.float32)
    true_positive=y_true+y_pred
    true_positive=true_positive.tolist()
    true_positive=Counter(true_positive)
    total_positives=Counter(y_true.tolist())

    return true_positive[2]/(total_positives[1]+0.000000000001)


    