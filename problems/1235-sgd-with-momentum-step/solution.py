import torch

def momentum_step(w, grad, v, lr, mu):
    """One SGD-with-momentum step.

    Args:
        w: parameter tensor
        grad: gradient tensor (same shape as w)
        v: velocity tensor (same shape as w)
        lr: learning rate (float)
        mu: momentum coefficient (float)

    Returns:
        (w_new, v_new) tuple of tensors
    """
    # TODO: v_new = mu * v + grad; w_new = w - lr * v_new
    x=torch.as_tensor(w,dtype=torch.float32)
    grad=torch.as_tensor(grad,dtype=torch.float32)
    v=torch.as_tensor(v,dtype=torch.float32)

    v=mu*v+grad
    w=w-lr*v
    return (w,v)
