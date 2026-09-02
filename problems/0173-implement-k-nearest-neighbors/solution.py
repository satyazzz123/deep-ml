import torch

def k_nearest_neighbors(points: torch.Tensor, query_point: torch.Tensor, k: int) -> torch.Tensor:
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: Tensor of shape (n, d) representing n points in d-dimensional space
        query_point: Tensor of shape (d,) representing the query point
        k: Number of nearest neighbors to return
    
    Returns:
        Tensor of shape (k, d) containing the k nearest neighbor points
        When distances are tied, points appearing earlier in the input tensor come first.
    """
    points=torch.as_tensor(points,dtype=torch.float32)
    query_point=torch.torch.as_tensor(query_point,dtype=torch.float32)
    distance=(torch.sum((points-query_point)**2,dim=-1,keepdims=True))
    distance= distance.squeeze(-1)
    idx=torch.sort(distance,descending=False).indices
    pred=idx[:k]
    return points[pred]