import torch
from collections import Counter
import math
def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculates the precision metric for binary classification.
    
    Precision is defined as the ratio of true positives to the sum of 
    true positives and false positives.
    
    Args:
        y_true: True binary labels (1D tensor)
        y_pred: Predicted binary labels (1D tensor)
    
    Returns:
        Precision value as a scalar tensor
    """
    # Your implementation here
    y_true=torch.as_tensor(y_true,dtype=torch.float32)
    y_pred=torch.as_tensor(y_pred,dtype=torch.float32)

    TP=y_true+y_pred
    FP=y_pred/y_true
    TP_count=Counter(TP.tolist())
    FP_count=Counter(FP.tolist())
    val=TP_count[2]/((FP_count[math.inf]+TP_count[2])+0.000000000001)

    return torch.round(torch.tensor(val),decimals=4)




