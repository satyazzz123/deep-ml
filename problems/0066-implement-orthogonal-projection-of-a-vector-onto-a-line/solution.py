import torch

def orthogonal_projection(v: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Compute the orthogonal projection of vector v onto line L using PyTorch.

    :param v: The vector to be projected (torch.Tensor)
    :param L: The line vector defining the direction of projection (torch.Tensor)
    :return: torch.Tensor representing the projection of v onto L, rounded to 3 decimal places
    """
    v=torch.as_tensor(v,dtype=torch.float32)
    L=torch.as_tensor(L,dtype=torch.float32)
    dot=torch.dot(v,L)
    # magnitude_v = torch.linalg.vector_norm(v)
    magnitude_L = torch.linalg.vector_norm(L)
    theta=dot/(magnitude_L*magnitude_L)


    return theta*L
