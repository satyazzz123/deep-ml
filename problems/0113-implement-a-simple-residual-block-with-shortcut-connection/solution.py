import torch
import torch.nn.functional as F
def residual_block(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """
    Implement a simple residual block with shortcut connection.
    
    Args:
        x: 1D input tensor
        w1: First weight matrix
        w2: Second weight matrix
    
    Returns:
        Output tensor after residual block processing
    """
    x=torch.as_tensor(x,dtype=torch.float32)
    w1=torch.as_tensor(w1,dtype=torch.float32)
    w2=torch.as_tensor(w2,dtype=torch.float32)
    
    y1=F.relu(x@w1)
    y2=(y1@w2)
    return F.relu(x+y2)
    


    