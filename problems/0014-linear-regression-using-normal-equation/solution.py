import torch

def linear_regression_normal_equation(X, y) -> torch.Tensor:
    """
    Solve linear regression via the normal equation using PyTorch.
    X: Tensor or convertible of shape (m,n); y: shape (m,) or (m,1).
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    x = torch.as_tensor(X, dtype=torch.float)
    y = torch.as_tensor(y, dtype=torch.float).reshape(-1,1)
    
    X_transpose=x.transpose(0,1)
    X_transpose_dot_X=torch.matmul(X_transpose,x)
    X_transpose_dot_X_inverse=torch.linalg.inv(X_transpose_dot_X)
    X_transpose_dot_y=torch.matmul(X_transpose,y)
    theta=torch.matmul(X_transpose_dot_X_inverse,X_transpose_dot_y)
    theta=torch.round(theta,decimals=4)
    return theta
