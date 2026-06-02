from re import T
import torch

def ridge_loss(X: torch.Tensor, w: torch.Tensor, y_true: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Implements the Ridge Regression Loss Function using PyTorch.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        w: Weight vector of shape (n_features,)
        y_true: True target values of shape (n_samples,)
        alpha: Regularization parameter (lambda)
    
    Returns:
        The Ridge loss value as a scalar tensor
    """
    # Your implementation here
    X=torch.as_tensor(X,dtype=torch.float32)
    w=torch.as_tensor(w,dtype=torch.float32)
    y_true=torch.as_tensor(y_true,dtype=torch.float32)
    pred_labels=X@w.T
    MSE=torch.mean((pred_labels-y_true)**2)
    regularization=alpha*(torch.sum(w**2))
    return (MSE+regularization)

