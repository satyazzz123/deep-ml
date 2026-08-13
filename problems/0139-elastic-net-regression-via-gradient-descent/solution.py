import torch
import torch.nn as nn
def elastic_net_gradient_descent(
    X: torch.Tensor,
    y: torch.Tensor,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    X=torch.as_tensor(X,dtype=torch.float32)
    batch_size,num_features=X.shape
    y=torch.as_tensor(y,dtype=torch.float32)
    w=torch.zeros(num_features,1,requires_grad=True)
    b=torch.zeros(1,requires_grad=True)
    y=y.reshape(-1,1)
    mse_fn = nn.MSELoss()

    for _ in range(max_iter):
      y_pred=X@w+b
      loss=0.5*mse_fn(y_pred,y)+alpha1*torch.sum(torch.abs(w))+alpha2*torch.sum(torch.square(w))
      loss.backward()
      converged = (torch.sum(torch.abs(w.grad)) < tol)

      if converged:
        break
      with torch.no_grad():
        w -= learning_rate * w.grad
        b-=learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()
    return(w.reshape(num_features),b)
      
