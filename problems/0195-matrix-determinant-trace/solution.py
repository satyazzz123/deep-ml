import torch

def matrix_determinant_and_trace(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the determinant and trace of a square matrix.
    
    Args:
        matrix: A square matrix (n x n) as a torch.Tensor
    
    Returns:
        Tuple of (determinant, trace) as torch.Tensors
    """
    # Your code here
    matrix=torch.as_tensor(matrix,dtype=torch.float32)
    trace=torch.sum(torch.diagonal(matrix,offset=0))

    return (torch.linalg.det(matrix),trace)