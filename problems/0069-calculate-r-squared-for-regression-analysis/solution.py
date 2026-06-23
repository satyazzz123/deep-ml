import torch

def r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the R-squared (RÂ²) coefficient of determination using PyTorch.

    Args:
        y_true (torch.Tensor): Tensor of true values
        y_pred (torch.Tensor): Tensor of predicted values

    Returns:
        float: R-squared value rounded to 3 decimal places
    """
    y_true=torch.as_tensor(y_true,dtype=torch.float32)
    y_pred=torch.as_tensor(y_pred,dtype=torch.float32)
    SS_res=torch.sum(torch.square(y_true-y_pred))
    SS_tot=torch.sum(torch.square(y_true-torch.mean(y_pred)))
    R2=1-(SS_res/(SS_tot+0.00000001))
    return torch.round(R2,decimals=3).item()