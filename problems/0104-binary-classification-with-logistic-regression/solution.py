import torch

def predict_logistic(X: torch.Tensor, weights: torch.Tensor, bias: float) -> torch.Tensor:
    """
    Implements binary classification prediction using Logistic Regression.

    Args:
        X: Input feature matrix (shape: N x D)
        weights: Model weights (shape: D)
        bias: Model bias

    Returns:
        Binary predictions (0 or 1)
    """
    # Your code here
    X=torch.as_tensor(X,dtype=torch.float32)
    w=torch.as_tensor(weights,dtype=torch.float32).reshape(1,-1)
    logits=X@w.T+bias
    probs=torch.sigmoid(logits)
    probs=torch.squeeze(probs,dim=1)
  
    return (probs >= 0.5).float() 
