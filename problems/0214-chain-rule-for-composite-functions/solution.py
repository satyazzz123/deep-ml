import torch

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.
    
    Args:
        functions: List of function names (applied right to left)
                   Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative
    
    Returns:
        Derivative value at x
    
    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    
    # Apply functions from right to left
    out = x
    for func_name in reversed(functions):
        if func_name == 'square':
            out = out ** 2
        elif func_name == 'sin':
            out = torch.sin(out)
        elif func_name == 'exp':
            out = torch.exp(out)
        elif func_name == 'log':
            out = torch.log(out)
        else:
            raise ValueError(f"Unsupported function: {func_name}")
            
    # Compute the derivative with respect to x
    out.backward()
    
    return x.grad.item()