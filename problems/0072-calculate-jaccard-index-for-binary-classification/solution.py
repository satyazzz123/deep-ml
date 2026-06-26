import torch
from collections import Counter
def jaccard_index(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the Jaccard Index for binary classification.

    Args:
        y_true: Binary tensor of true labels.
        y_pred: Binary tensor of predicted labels.

    Returns:
        Jaccard Index as a float rounded to 3 decimal places.
    """
    # Write your code here
    y_true=torch.as_tensor(y_true,dtype=torch.float32)
    y_pred=torch.as_tensor(y_pred,dtype=torch.float32)

    TP=y_true/y_pred
    TP=Counter(TP.tolist())
    TP=TP[1]

    FP=y_true/y_pred
    FP=Counter(FP.tolist())
    FP=FP[0]

    FN=y_pred/y_true
    FN=Counter(FN.tolist())
    FN=FN[0]


    return round((TP/(TP+FP+FN)),ndigits=3)


