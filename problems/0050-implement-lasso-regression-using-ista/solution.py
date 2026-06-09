import torch

def soft_threshold(w: torch.Tensor, threshold: float) -> torch.Tensor:
    """Apply soft-thresholding operator element-wise.
    
    S(w, Î») = sign(w) * max(|w| - Î», 0)
    
    Args:
        w: Input tensor
        threshold: Threshold value Î»
    
    Returns:
        Soft-thresholded tensor where:
        - Values with |w| > Î» are shrunk toward zero by Î»
        - Values with |w| â¤ Î» become exactly zero
    """
    # Your code here
    return torch.sign(w) * torch.clamp(torch.abs(w)-threshold,min=0)



def l1_regularization_gradient_descent(X: torch.Tensor, y: torch.Tensor, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).
    
    ISTA alternates between:
    1. Gradient step on MSE loss: w_temp = w - lr * gradient_mse
    2. Proximal step (soft-thresholding): w_new = soft_threshold(w_temp, lr * alpha)
    
    Args:
        X: Feature matrix tensor of shape (n_samples, n_features)
        y: Target vector tensor of shape (n_samples,)
        alpha: L1 regularization strength
        learning_rate: Step size for gradient descent
        max_iter: Maximum iterations
        tol: Convergence tolerance on weight change
    
    Returns:
        tuple: (weights, bias) as torch.Tensors
    
    Note: The bias term is NOT regularized.
    """
    n_samples, n_features = X.shape
    X=torch.as_tensor(X,dtype=torch.float32)
    y=torch.as_tensor(y,dtype=torch.float32)
    weights = torch.zeros(n_features, dtype=torch.float32,requires_grad=True)
    bias = torch.tensor(0.0, dtype=torch.float32,requires_grad=True)
    
    # Your code herea
    for _ in range(max_iter):
      pred=X@weights +bias
      loss_1=torch.mean(torch.square(pred-y))/2
      loss_1.backward()
      with torch.no_grad():
        old_weights = weights.clone()
        weights-=learning_rate*weights.grad
        bias-=learning_rate*bias.grad

        weights.copy_(soft_threshold(weights,learning_rate*alpha))
        weights.grad.zero_()
        bias.grad.zero_()
        if torch.norm(weights - old_weights) < tol:
          break
    return (weights.detach(),bias.detach())

      

    
