import torch
import math
def optimizer_part(w,grad,beta1,beta2,epsilon,learning_rate,m_t_1=0,v_t_1=0):
  # m_t_1=[0]
  m_t=(beta1*m_t_1)+(1-beta1)*grad
  # m_t_1.append(m_t)
  # v_t_1=[0]
  v_t=(beta2*v_t_1)+(1-beta2)*grad**2
  # v_t_1.append(v_t)
  m_cap=m_t/(1-beta1)
  v_cap=v_t/(1-beta2)
  w-=(learning_rate*m_cap)/(v_cap**(0.5)+epsilon)
  return w,m_t_1,v_t_1




def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10) -> torch.Tensor:
    """
    Implements Adam optimization algorithm using PyTorch's built-in optimizer.

    Args:
        f: The objective function to be optimized
        grad: A function that computes the gradient (unused; autograd is used instead)
        x0: Initial parameter values (torch.Tensor)
        learning_rate: The step size (default: 0.001)
        beta1: Exponential decay rate for the first moment estimates (default: 0.9)
        beta2: Exponential decay rate for the second moment estimates (default: 0.999)
        epsilon: A small constant for numerical stability (default: 1e-8)
        num_iterations: Number of iterations to run the optimizer (default: 10)

    Returns:
        torch.Tensor: Optimized parameters
    """
    # Your code here
    m_t_1=0.0
    v_t_1=0.0
    x0=torch.as_tensor(x0,dtype=torch.float32)
    for _ in range(num_iterations):
      val=f(x0)
      grad_vals=grad(x0)
      # m_t_1=0
      # v_t_1=0
      x0,m_t_1,v_t_1=optimizer_part(x0,grad_vals,beta1,beta2,epsilon,learning_rate,m_t_1,v_t_1) #optimizer
      grad_vals=0
    return x0
