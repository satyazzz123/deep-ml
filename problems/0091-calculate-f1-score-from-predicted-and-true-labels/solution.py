import torch
from collections import Counter
def calculate_f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the F1 score based on true and predicted labels using PyTorch.

    Args:
        y_true (torch.Tensor): True labels (ground truth).
        y_pred (torch.Tensor): Predicted labels.

    Returns:
        float: The F1 score rounded to three decimal places.
    """
    # Your code here
    y_true=torch.as_tensor(y_true,dtype=torch.float32)
    y_pred=torch.as_tensor(y_pred,dtype=torch.float32)

    TP=y_true+y_pred
    count=Counter(TP.tolist())
    TP=count[2]
    # print(TP)
    FP=y_true/y_pred
    count=Counter(FP.tolist())
    FP=count[0]
    FN=y_pred/y_true
    count=Counter(FN.tolist())
    FN=count[0]

    f1=(2*TP)/(2*TP+FP+FN)
    return round(f1,ndigits=3)



