import torch
from typing import Union

def accuracy_score(y_true: Union[torch.Tensor, list, "np.ndarray"],
                   y_pred: Union[torch.Tensor, list, "np.ndarray"]) -> float:
    """
    Compute the accuracy: fraction of matching elements in y_true and y_pred.
    Both inputs may be torch.Tensor, list, or numpy.ndarray.
    """
    # Your implementation here
    true_labels=0
    for i in range(len(y_true)):
       if y_true[i]==y_pred[i]:
        true_labels+=1
    return true_labels/(len(y_true))


