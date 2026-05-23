import torch

def kernel_function(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Computes the linear kernel between two input vectors.
    The linear kernel is defined as the dot product (inner product) of two vectors.
    
    Args:
        x1: First input tensor (1D vector)
        x2: Second input tensor (1D vector)
    
    Returns:
        Scalar tensor representing the linear kernel (dot product)
    """
    # Your implementation here
    # x1=torch.as_tensor(x1)
    # x2=torch.as_tensor(x2)

    dot_product=0
    for i in range(len(x1)):
        dot_product+=x1[i]*x2[i]
    return torch.tensor(dot_product)

