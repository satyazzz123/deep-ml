import torch

def confusion_matrix(data: list) -> torch.Tensor:
    """
    Generate a 2x2 confusion matrix for binary classification.

    Args:
        data: A list of [y_true, y_pred] pairs for binary labels (0 or 1)

    Returns:
        A 2x2 torch.Tensor confusion matrix arranged as [[TP, FN], [FP, TN]]
    """
    TP=0
    FP=0
    TN=0
    FN=0
    for i in data:
      if (i[0]+i[1]==2):
        TP+=1
      if (i[0]-i[1]==-1):
        FP+=1
      if (i[0]+i[1]==0):
        TN+=1
      if (i[1]-i[0]==-1):
        FN+=1
    return torch.tensor([[TP,FN],[FP,TN]])
      
