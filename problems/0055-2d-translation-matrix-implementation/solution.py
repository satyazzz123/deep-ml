import torch

def translate_object(points, tx, ty) -> torch.Tensor:
    """
    Apply a 2D translation matrix to a set of points.
    
    Args:
        points: list of [x, y] coordinates or torch.Tensor of shape (N, 2)
        tx: translation distance in x direction
        ty: translation distance in y direction
    
    Returns:
        torch.Tensor of translated points with shape (N, 2)
    """
    translation_matrix=torch.tensor([tx,ty],dtype=torch.float32)
    points=torch.tensor(points,dtype=torch.float32)


    return (points+translation_matrix)
