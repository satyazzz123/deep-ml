import torch
from collections import Counter
def f_score(y_true: torch.Tensor, y_pred: torch.Tensor, beta: float) -> float:
    """
    Calculate F-Score for a binary classification task.

    :param y_true: torch.Tensor of true labels (binary)
    :param y_pred: torch.Tensor of predicted labels (binary)
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """
    y_true=torch.as_tensor(y_true,dtype=torch.float32)
    y_pred=torch.as_tensor(y_pred,dtype=torch.float32)
    TP=y_true+y_pred
    TP=Counter(TP.tolist())
    TP=TP[2]
    FP=y_true/y_pred
    FP=Counter(FP.tolist())
    FP=FP[0]
    FN=y_pred/y_true
    FN=Counter(FN.tolist())
    FN=FN[0]
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1=((1+beta**2)*(precision*recall))/(((beta**2) * precision)+recall)

    return F1
