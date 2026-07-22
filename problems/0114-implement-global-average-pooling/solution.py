import torch

def global_avg_pool(x: torch.Tensor) -> torch.Tensor:
    """
    Performs Global Average Pooling on a 3D tensor representing feature maps.
    
    Args:
        x: Input tensor of shape (height, width, channels)
    
    Returns:
        1D tensor of shape (channels,) with average values per channel
    """
    x=torch.as_tensor(x,dtype=torch.float32)
    H,W,C=x.shape
    x=torch.sum(x,dim=1)
    print(x)
    x=torch.sum(x,dim=0)
    print(x)
    return x/(H*W)
