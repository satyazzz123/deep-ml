import numpy as np

def prelu_forward(x: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """
    Implements the forward pass of PReLU.
    Args:
        x: Input array of any shape
        alpha: Slope parameter for negative values (default: 0.25)
    Returns:
        np.ndarray: PReLU activation output, same shape as x
    """
    # Your code here
    prelu_tensor=[]
    for i in range(len(x)):
      if x[i]>0:
        prelu_tensor.append(x[i])
      else:
        prelu_tensor.append(x[i]*alpha)
    return np.array(prelu_tensor)


def prelu_backward(x: np.ndarray, alpha: float, grad_output: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Implements the backward pass of PReLU, computing gradients for both x and alpha.
    Args:
        x: Original input from forward pass
        alpha: Slope parameter used in forward pass
        grad_output: Upstream gradient, same shape as x
    Returns:
        grad_x: Gradient w.r.t. input x, same shape as x
        grad_alpha: Gradient w.r.t. alpha (scalar, summed over all elements)
    """
    prelu_tensor=[]
    for i in range(len(x)):
      if x[i]>0:
        prelu_tensor.append(0)
      else:
        prelu_tensor.append(x[i])
    prelu_tensor=np.array(prelu_tensor)
    grad=prelu_tensor*grad_output
    grad_value=np.sum(grad)
    prelu_tensor=[]
    for i in range(len(x)):
      if x[i]>0:
        prelu_tensor.append(1)
      else:
        prelu_tensor.append(alpha)
    prelu_tensor=np.array(prelu_tensor)
    return (prelu_tensor*grad_output,grad_value.item())





    