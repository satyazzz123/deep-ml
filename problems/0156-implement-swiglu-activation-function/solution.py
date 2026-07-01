import torch

def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor of shape (batch_size, 2d)
    Returns:
        torch.Tensor of shape (batch_size, d)
    """
    # Your code here
    x=torch.as_tensor(x,dtype=torch.float32)
    shape=x.shape
    value=[]
    half=(shape[1])//2
    # print(half)
    for i in x:
      x1=i[:half]
      x2=i[half:]
      x3=torch.sigmoid(x2)
      swish=x3*x2
      x4=x1*swish
      value.append(x4)
    return torch.stack(value)
