import torch

def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error between two tensors.

    Parameters:
        y_true (torch.Tensor): Tensor of true values
        y_pred (torch.Tensor): Tensor of predicted values

    Returns:
        float: Mean Absolute Error
    """
    # Your code here
    y_true=torch.as_tensor(y_true,dtype=torch.float32).reshape(1,-1)
    y_pred=torch.as_tensor(y_pred,dtype=torch.float32).reshape(1,-1)
    diff=y_pred-y_true
    abs=torch.abs(diff)
    return (torch.sum(abs,dim=-1)/len(y_true[0])).item()