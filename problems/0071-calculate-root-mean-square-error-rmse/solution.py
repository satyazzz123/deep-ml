import torch
import torch.nn.functional as F

def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate Root Mean Square Error (RMSE) between actual and predicted values.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        RMSE value rounded to three decimal places.
    """
    # Write your code here
    y_true=torch.as_tensor(y_true,dtype=torch.float32)
    y_pred=torch.as_tensor(y_pred,dtype=torch.float32)
    rmse_val=torch.sum(torch.square(y_true-y_pred))/len(y_true)
    rmse_val=rmse_val.item()
    return round(rmse_val**(0.5),ndigits=3)
