import torch

def compute_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int, normalize=None, round_decimals=4) -> torch.Tensor:
    """
    Compute a KxK confusion matrix with optional normalization.

    Args:
        y_true: Tensor of true labels in [0, K-1]
        y_pred: Tensor of predicted labels in [0, K-1]
        num_classes: K, number of classes
        normalize: None | 'true' | 'pred' | 'all'
        round_decimals: decimals to round when normalization is applied

    Returns:
        torch.Tensor confusion matrix
    """
    # Your implementation here
    y_true=torch.as_tensor(y_true)
    y_pred=torch.as_tensor(y_pred)

    confusion_matrix=torch.zeros((num_classes,num_classes))
    for i in range(len(y_true)):
      confusion_matrix[y_true[i],y_pred[i]]+=1

    if normalize=="true":
      confusion_matrix=confusion_matrix/torch.sum(confusion_matrix,dim=-1,keepdims=True)
    elif normalize=="pred":
      confusion_matrix=confusion_matrix/torch.sum(confusion_matrix,dim=0,keepdims=True)
    elif normalize=="all":
      confusion_matrix=confusion_matrix/torch.sum(confusion_matrix,dim=(-1,1))





    return torch.round(confusion_matrix,decimals=round_decimals)

