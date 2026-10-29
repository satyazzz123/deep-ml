import torch

def softmax_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Jacobian matrix of the softmax function using PyTorch.
    
    Args:
        x: Input tensor
        
    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    # Your code here - you can use torch.autograd.functional.jacobian
    # or compute it directly from softmax output
    x=torch.as_tensor(x,dtype=torch.float32)
    jacobian=torch.zeros((x.shape[0],x.shape[0]))
    softmax_values=torch.softmax(x,dim=-1)
    for i in range(len(jacobian[0])):
      for j in range(len(jacobian[0])):
        if i==j:
          jacobian[i][j]=softmax_values[i]*(1-softmax_values[i])
        else:
          jacobian[i][j]=-1*softmax_values[i]*softmax_values[j]
        
    return jacobian
