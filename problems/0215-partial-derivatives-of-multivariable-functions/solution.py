import torch

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> torch.Tensor:
    """
    Compute partial derivatives of multivariable functions using PyTorch autograd.
    
    Args:
        func_name: Function identifier
            'poly2d': f(x,y) = x^2*y + x*y^2
            'exp_sum': f(x,y) = e^(x+y)
            'product_sin': f(x,y) = x*sin(y)
            'poly3d': f(x,y,z) = x^2*y + y*z^2
            'squared_error': f(x,y) = (x-y)^2
        point: Point (x, y) or (x, y, z) at which to evaluate
    
    Returns:
        Tensor of partial derivatives [df/dx, df/dy, ...] at point
    """
    # Convert point to a tensor with gradient tracking enabled
    x = torch.tensor(point, dtype=torch.float32, requires_grad=True)
    
    if func_name == 'poly2d':
        x_val, y_val = x[0], x[1]
        f = (x_val ** 2) * y_val + x_val * (y_val ** 2)
    elif func_name == 'exp_sum':
        x_val, y_val = x[0], x[1]
        f = torch.exp(x_val + y_val)
    elif func_name == 'product_sin':
        x_val, y_val = x[0], x[1]
        f = x_val * torch.sin(y_val)
    elif func_name == 'poly3d':
        x_val, y_val, z_val = x[0], x[1], x[2]
        f = (x_val ** 2) * y_val + y_val * (z_val ** 2)
    elif func_name == 'squared_error':
        x_val, y_val = x[0], x[1]
        f = (x_val - y_val) ** 2
    else:
        raise ValueError(f"Unsupported function name: {func_name}")
        
    # Compute backward pass to calculate gradients
    f.backward()
    
    return x.grad